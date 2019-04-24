#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import itertools
import json
import os
import math
import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils, bleu
from fairseq.data import iterators
from fairseq.distributed_utils import all_reduce
from fairseq.fed_utils import save_expert_outputs
from fairseq.summary_writer import SummaryWriter
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)

    # Build trainer
    trainer = Trainer(args, task, model, criterion, dummy_batch)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    summary_writer = SummaryWriter(log_dir=args.save_dir, enable=args.distributed_rank == 0)

    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    first_train = True
    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        trainer.dummy_train_step([dummy_batch])
        first_train = False

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')

    if not hasattr(save_checkpoint, 'not_best'):
        save_checkpoint.not_best = 0

    if not args.no_first_valid and first_train:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, True, summary_writer)

    if args.finetune_params != '':
        print("| train parameters.")
        for name, param in trainer.model.named_parameters():
            if trainer.should_train(name):
                print(name)

        print("| fixed parameters.")
        for name, param in trainer.model.named_parameters():
            if not trainer.should_train(name):
                print(name)

    if args.start_ckpt != '':
        save_checkpoint.not_best = 0
        save_checkpoint.best = 9999

    print("| train begin.")
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr, summary_writer)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets,
                                    epoch_itr.epoch % args.test_bleu_interval == 0, summary_writer)
            if args.early_stop > 0:
                if hasattr(save_checkpoint, 'best') and valid_losses[0] > save_checkpoint.best:
                    save_checkpoint.not_best += 1
                    print("| Not the best ckpt... not best:", save_checkpoint.not_best)
                    if save_checkpoint.not_best > args.early_stop:
                        print("| Early stop...")
                        break
                else:
                    save_checkpoint.not_best = 0

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))
    os.system("ps aux | grep redis-server | awk '{print $2}' | xargs kill")

    if args.save_output:
        save_expert_outputs(args, task, trainer)


def train(args, trainer, task, epoch_itr, summary_writer=None):
    """Train the model for one epoch."""

    # Update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus)
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)

    distributed_utils.barrier(args, "train_%d" % trainer.get_num_updates())
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg

        stats['progress'] = round(i / num_batches * args.distributed_world_size * args.update_freq[-1], 3)
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
            distributed_utils.barrier(args, "train_val_%d" % trainer.get_num_updates())

        if num_updates % args.log_interval == 0:
            summary_writer.log_stats('train', stats, num_updates)

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = round(trainer.get_meter('train_wall').sum)
    return stats


def validate(args, trainer, task, epoch_itr, subsets, test_bleu=False, summary_writer=None):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []

    distributed_utils.barrier(args, "validate1_%d" % trainer.get_num_updates())
    for subset in subsets:
        # Initialize data iterator
        def get_itr():
            itr = task.get_batch_iterator(
                dataset=task.dataset(subset),
                max_tokens=args.max_tokens,
                max_sentences=args.max_sentences_valid,
                max_positions=utils.resolve_max_positions(
                    task.max_positions(),
                    trainer.get_model().max_positions(),
                ),
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=8,
                seed=args.seed,
                num_shards=args.distributed_world_size,
                shard_id=args.distributed_rank,
            ).next_epoch_itr(shuffle=False)
            progress = progress_bar.build_progress_bar(
                args, itr, epoch_itr.epoch,
                prefix='valid on \'{}\' subset'.format(subset),
                no_progress_bar='simple'
            )
            return progress

        progress = get_itr()

        num_dataset = task.dataset(subset).num_dataset

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)
            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        bleu_scorers = [bleu.Scorer(
            task.target_dictionary.pad(),
            task.target_dictionary.eos(),
            task.target_dictionary.unk()
        ) for _ in range(num_dataset)] if test_bleu else None

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg

        if bleu_scorers is not None:
            # test bleu
            print("| test bleu.")
            sample_size = [0 for _ in range(num_dataset)]
            bleu_scores = [0 for _ in range(num_dataset)]
            progress = get_itr()

            tgt_str_files = []
            hypo_str_files = []
            for ds_id in range(num_dataset):
                tgt_str_path = task.dataset(subset).dataset_names[ds_id] + '.tgt.txt'
                hypo_str_path = task.dataset(subset).dataset_names[ds_id] + '.hypo.txt'
                tgt_str_files.append(open(os.path.join(args.save_dir, tgt_str_path), 'w', encoding='utf-8'))
                hypo_str_files.append(open(os.path.join(args.save_dir, hypo_str_path), 'w', encoding='utf-8'))

            def print_to_file(dataset_id, tgt_str, hypo_str):
                tgt_str_files[dataset_id].write(tgt_str + '\n')
                hypo_str_files[dataset_id].write(hypo_str + '\n')

            for sample in progress:
                trainer.test_bleu_step(sample, bleu_scorers, print_to_file)
                if 'dataset_id' in sample:
                    for ds_id in range(num_dataset):
                        sample_size[ds_id] += (sample['dataset_id'] == ds_id).int().sum().item()
                elif 'id' in sample:
                    sample_size[0] += len(sample['id'])

            for f in tgt_str_files + hypo_str_files:
                f.close()

            distributed_utils.barrier(args, "validate2_%d" % trainer.get_num_updates())
            for ds_id in range(num_dataset):
                try:
                    bleu_scores[ds_id] = bleu_scorers[ds_id].score() * sample_size[ds_id]
                except Exception as e:
                    bleu_scores[ds_id] = 0

            sample_size = torch.Tensor(sample_size).cuda()
            bleu_scores = torch.Tensor(bleu_scores).cuda()
            if args.distributed_world_size > 1:
                all_reduce(sample_size)
                all_reduce(bleu_scores)

            bleu_dict = {}
            for ds_id in range(num_dataset):
                if sample_size[ds_id].item() > 0:
                    name = "bleu_" + task.dataset(subset).dataset_names[ds_id]
                    bleu_dict[name] = stats[name] = bleu_scores[ds_id].item() / sample_size[ds_id].item()
                    try:
                        train_ds_id = task.dataset('train').dataset_names.index(
                            task.dataset(subset).dataset_names[ds_id])
                        task.dataset('train').student_scores[train_ds_id] = bleu_dict[name]
                    except ValueError:
                        pass
            output_path = os.path.join(args.save_dir, 'val_bleu.json')
            json.dump(bleu_dict, open(output_path, 'w'))

        progress.print(stats)
        if summary_writer is not None:
            summary_writer.log_stats('val/' + subset, stats, trainer.get_num_updates())

        valid_losses.append(stats['valid_loss'])
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)

    if args.no_save or not distributed_utils.is_master(args):
        return

    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
        'not_best': getattr(save_checkpoint, 'not_best', 0),
    }

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt') + \
                      utils.checkpoint_paths(args.save_dir)
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    if args.start_ckpt == '':
        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    else:
        checkpoint_path = args.start_ckpt

    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              eval(args.optimizer_overrides))
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'], args.reproduce,
                                      not args.reproduce and args.fix_batches_to_gpus)

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
            if 'not_best' in extra_state:
                save_checkpoint.not_best = extra_state['not_best']
        return True
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        task.load_dataset(split, combine=True)
        # if split == 'train':
        # task.load_dataset(split, combine=True)
        # else:
        #     for k in itertools.count():
        #         split_k = split + (str(k) if k > 0 else '')
        #         try:
        #             task.load_dataset(split_k, combine=False)
        #         except FileNotFoundError as e:
        #             if k > 0:
        #                 break
        #             raise e


if __name__ == '__main__':
    parser = options.get_training_parser('universal_translation')
    parser.add_argument('--save-output', action='store_true')
    parser.add_argument('--early-stop', default=10, type=int)
    parser.add_argument('--distill-topk', default=4, type=int)
    parser.add_argument('--no-first-valid', action='store_true')
    parser.add_argument('--test-bleu-interval', default=3, type=int)
    parser.add_argument('--reproduce', action='store_true')
    parser.add_argument('--finetune-params', default='', type=str)
    parser.add_argument('--finetune-params-exclude', default='', type=str)
    parser.add_argument('--start-ckpt', default='', type=str)
    parser.add_argument('--universal', action='store_true')
    parser.add_argument('--data-limit', default='', type=str)
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)

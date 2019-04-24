import glob
import hashlib
import os
import torch
from tqdm import tqdm
from fairseq import utils, distributed_utils
import numpy as np
import ujson as json

from fairseq.data.indexed_dataset import IndexedDatasetBuilder, IndexedCachedDataset

FED_VERSION_FN = 'fed_version.v3.idx'


def dist2topk(out_dist, k):
    topk_prob, topk_idx = torch.topk(out_dist, k, dim=-1)
    topk_prob = topk_prob.view(-1, k)  # (B x T) x k
    topk_prob = topk_prob / topk_prob.sum(1, keepdim=True)
    topk_idx = topk_idx.view(-1, k)  # (B x T) x k
    return topk_idx, topk_prob


def output2topk(output, k):
    topk_outp, topk_idx = torch.topk(output, k, dim=-1)
    topk_outp = topk_outp.view(-1, k)  # (B x T) x k
    topk_idx = topk_idx.view(-1, k)  # (B x T) x k
    return topk_idx, topk_outp


def get_sample_key(ids):
    if not hasattr(get_sample_key, 'sample_key_cache'):
        get_sample_key.sample_key_cache = {}
    ids_str = ','.join([str(id) for id in sorted(ids)])
    if ids_str not in get_sample_key.sample_key_cache:
        hash_object = hashlib.md5(ids_str.encode())
        get_sample_key.sample_key_cache[ids_str] = hash_object.hexdigest()
    return get_sample_key.sample_key_cache[ids_str]


class TeacherOutputDatasetBuilder(IndexedDatasetBuilder):
    def add_item(self, data):
        # +1 for Lua compatibility
        data = np.array(data, dtype=self.dtype)
        bytes = self.out_file.write(data)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in data.shape:
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(data.shape))


class TeacherOutputDataset(IndexedCachedDataset):
    dtype2size = {
        float: 8,
        int: 4,
    }

    def __init__(self, prefix):
        self.cache_index = {}
        super().__init__(prefix, fix_lua_indexing=False)

    @staticmethod
    def save_bin(prefix, data_list, dtype=np.float):
        bin_path = prefix + '.bin'
        idx_path = prefix + '.idx'
        builder = TeacherOutputDatasetBuilder(bin_path, dtype)
        for d in data_list:
            builder.add_item(d)
        builder.finalize(idx_path)

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)

        if i in self.cache:
            np.copyto(a, self.cache[i])
        else:
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            self.cache[i] = a

        item = torch.from_numpy(a)
        if self.dtype == np.int32 or self.dtype == np.int or self.dtype == np.int64:
            item = item.long()
        else:
            item = item.float()
        return item


def gen_outputs(args, task, trainer):
    trainer.model.eval()
    itr = task.get_batch_iterator(
        dataset=task.dataset('train'),
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

    outputs = [None for _ in range(len(task.dataset('train')))]
    for sample in tqdm(itr, mininterval=5):
        with torch.no_grad():
            if sample is None or len(sample) == 0:
                continue
            sample = utils.move_to_cuda(sample)

            bs, srclen = sample['net_input']['src_tokens'].shape
            output = trainer.model(**sample['net_input'])[0].detach()
            non_padding_mask = sample['target'].ne(task.target_dictionary.pad()).cpu()
            _, tgtlen = sample['target'].shape
            topk_idx, topk_v = output2topk(output, args.distill_topk)
            topk_x_shape = (bs, tgtlen, args.distill_topk)
            topk_idx, topk_v = topk_idx.view(*topk_x_shape).cpu().numpy(), topk_v.view(*topk_x_shape).cpu().numpy()
            non_padding_mask = non_padding_mask.view(*topk_x_shape[:2]).cpu().numpy().astype(bool)
            for b in range(bs):
                outputs[sample['id'][b].item()] = \
                    topk_idx[b, non_padding_mask[b]].tolist(), \
                    topk_v[b, non_padding_mask[b]].tolist()
    return outputs


def save_expert_outputs(args, task, trainer):
    print("| Start saving expert outputs..")
    expert_outputs = gen_outputs(args, task, trainer)
    output_path = os.path.join(args.save_dir, 'train_output.json.{}'.format(args.distributed_rank))
    json.dump(expert_outputs, open(output_path, 'w'))
    distributed_utils.barrier(args, 'save_expert_outputs')
    if distributed_utils.is_master(args):
        expert_outputs_ = []
        val_bleu_path1 = os.path.join(args.save_dir, 'val_bleu.json')
        val_bleu_path2 = os.path.join(args.data[0], 'expert_bleu_{}_{}.json'.format(args.sources, args.targets))
        os.system('cp {} {}'.format(val_bleu_path1, val_bleu_path2))

        for i in range(args.distributed_world_size):
            output_path = os.path.join(args.save_dir, 'train_output.json.{}'.format(i))
            expert_outputs_.append(json.load(open(output_path, 'r')))
            try:
                os.remove(output_path)
            except:
                pass

        for j in range(len(expert_outputs_[0])):
            for i in range(args.distributed_world_size):
                if expert_outputs_[i][j] is not None:
                    expert_outputs[j] = expert_outputs_[i][j]
                    break
            assert expert_outputs[j] is not None

        path = os.path.join(args.data[0], '{}_{}_topk_idx'.format(args.sources, args.targets))
        TeacherOutputDataset.save_bin(path, [o[0] for o in expert_outputs], np.int32)

        path = os.path.join(args.data[0], '{}_{}_topk_prob'.format(args.sources, args.targets))
        TeacherOutputDataset.save_bin(path, [o[1] for o in expert_outputs], np.float)

    print("| Save expert@{}_{}".format(args.sources, args.targets))

# def save_master_outputs(args, task, trainer, version, dev_scores, force_save=False):
#     assert dev_scores is not None
#     master_outputs = None
#
#     try:
#         with open(os.path.join(args.fed_path, 'all_{}'.format(args.target_lang), FED_VERSION_FN)) as f:
#             old_version_data = json.load(f)
#     except:
#         old_version_data = None
#
#     dataset = task.dataset('train')
#     division = dataset.src_cumsum + [len(dataset)]
#     version_path = os.path.join(args.save_dir, FED_VERSION_FN)
#     version_data = {
#         'version': version,
#     }
#
#     for lng_idx, lng in enumerate(dataset.fed_lngs):
#         start, end = division[lng_idx], division[lng_idx + 1]
#         if force_save or old_version_data is None or dev_scores['bleu_{}'.format(lng)] > old_version_data[
#             'bleu_{}'.format(lng)]:
#             output_path = os.path.join(args.save_dir, 'train_output.{}.json.{}'.format(lng, args.distributed_rank))
#             if master_outputs is None:
#                 master_outputs = gen_outputs(args, task, trainer)
#             json.dump(master_outputs[start:end], open(output_path, 'w'))
#             version_data['bleu_{}'.format(lng)] = dev_scores['bleu_{}'.format(lng)]
#         else:
#             version_data['bleu_{}'.format(lng)] = old_version_data['bleu_{}'.format(lng)]
#
#     if distributed_utils.is_master(args):
#         with open(version_path, 'w') as f:
#             json.dump(version_data, f)
#         print("| Save master, data:{}".format(json.dumps(version_data)))
#
#
# def load_master_outputs(args, score, old_master_version=None, old_master_outputs=None):
#     assert score is not None
#     master_outputs = old_master_outputs
#     master_version = old_master_version
#
#     files = glob.glob(os.path.join(args.fed_path, 'all_{}'.format(args.target_lang),
#                                    'train_output.{}.*.*'.format(args.source_lang)))
#     if len(files) == 0:
#         files = glob.glob(os.path.join(args.fed_path, 'train_output.{}.*.*'.format(args.source_lang)))
#         if len(files) == 0:
#             print("| Master not found.")
#             return master_version, master_outputs
#
#     try:
#         version_fn = os.path.join(args.fed_path, 'all_{}'.format(args.target_lang), FED_VERSION_FN)
#         if not os.path.exists(version_fn):
#             version_fn = os.path.join(args.fed_path, FED_VERSION_FN)
#         with open(version_fn) as f:
#             version_data = json.load(f)
#         version = version_data['version']
#
#         if old_master_version is not None and old_master_outputs is not None:
#             if version <= old_master_version:
#                 print("| Master has not updated yet.")
#                 return master_version, master_outputs
#     except FileNotFoundError:
#         print("| Master version not found.")
#         return master_version, master_outputs
#
#     outputs = []
#     for f in files:
#         outputs.append(json.load(open(f, 'r')))
#     outputs_flatten = [None for _ in range(len(outputs[0]))]
#     for i in range(len(outputs[0])):
#         for j in range(len(files)):
#             if outputs[j][i] is not None:
#                 outputs_flatten[i] = outputs[j][i]
#                 break
#         assert outputs_flatten[i] is not None
#     print("| Load master@{}.".format(version))
#     return version, outputs_flatten

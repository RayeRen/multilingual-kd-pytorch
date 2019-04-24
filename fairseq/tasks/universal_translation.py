# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import glob
import json
import os

import numpy as np

from fairseq import options
from fairseq.data import (
    data_utils, Dictionary, ConcatDataset,
    IndexedCachedDataset
)
from fairseq.data.universal_dataset import UniversalDataset
from fairseq.fed_utils import TeacherOutputDataset
from . import FairseqTask, register_task


@register_task('universal_translation')
class UniversalTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('--sources', default='', metavar='SRC',
                            help='source languages')
        parser.add_argument('--targets', default='', metavar='TARGET',
                            help='target languages')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        all_lngs = json.load(open(os.path.join(args.data[0], 'all_lngs.json')))
        self.id2lng = all_lngs
        self.lng2id = {v: k for k, v in enumerate(all_lngs)}

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        # if args.source_lang is None or args.target_lang is None:
        #     args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        # if args.source_lang is None or args.target_lang is None:
        #     raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        if args.share_all_embeddings:
            src_dict = Dictionary.load(os.path.join(args.data[0], 'dict.txt'))
            tgt_dict = Dictionary.load(os.path.join(args.data[0], 'dict.txt'))
        else:
            src_dict = Dictionary.load(os.path.join(args.data[0], 'dict.txt'))
            tgt_dict = Dictionary.load(os.path.join(args.data[0], 'dict.tgt.txt'))

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format('src', len(src_dict)))
        print('| [{}] dictionary: {} types'.format('tgt', len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def indexed_dataset(path, dictionary):
            return IndexedCachedDataset(path, fix_lua_indexing=True)

        src_datasets = []
        tgt_datasets = []
        src_lngs = []
        tgt_lngs = []
        dataset_ids = []
        dataset_names = []
        lng_borders = [0]
        data_path = self.args.data[0]
        fns = glob.glob(os.path.join(data_path, f'{split}.*'))
        lng_pairs = list(set([f.split('.')[1] for f in fns]))
        lng_pairs = sorted(lng_pairs)
        ds_idx = 0
        sources = [s for s in self.args.sources.split(",") if s != '']
        targets = [t for t in self.args.targets.split(",") if t != '']

        is_distill = self.args.criterion == 'distill_label_smoothed_cross_entropy' and split == 'train'
        topk_idxs = []
        topk_probs = []
        expert_scores = []

        for idx, lng_pair in enumerate(lng_pairs):
            src, tgt = lng_pair.split('-')
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split, src, tgt))

            def add_dataset(src, tgt):
                if (src not in sources and len(sources) > 0) or (tgt not in targets and len(targets) > 0):
                    return 0
                if not os.path.exists(prefix + src + ".bin") or \
                        not os.path.exists(prefix + tgt + ".bin"):
                    return 0

                if is_distill and not os.path.exists(
                        os.path.join(self.args.data[0], '{}_{}_topk_idx.idx'.format(src, tgt))):
                    return 0

                src_ds = indexed_dataset(prefix + src, self.src_dict)
                src_datasets.append(src_ds)
                tgt_ds = indexed_dataset(prefix + tgt, self.tgt_dict)
                tgt_datasets.append(tgt_ds)

                l = len(src_ds)
                if self.args.data_limit != '' \
                        and src + "-" + tgt == self.args.data_limit.split(':')[0] \
                        and l > int(self.args.data_limit.split(':')[1]):
                    l = int(self.args.data_limit.split(':')[1])
                    src_datasets[-1].size = l
                    tgt_datasets[-1].size = l
                    l = len(src_ds)

                print("| Add dataset {} -> {}. size:{}".format(src, tgt, l))
                lng_borders.append(lng_borders[-1] + l)
                dataset_names.append(f"{src}_{tgt}")
                for i in range(l):
                    src_lngs.append(self.lng2id[src])
                    tgt_lngs.append(self.lng2id[tgt])
                    dataset_ids.append(ds_idx)

                if is_distill:
                    assert self.args.data_limit == ''
                    path = os.path.join(self.args.data[0], '{}_{}_topk_idx'.format(src, tgt))
                    topk_idxs.append(TeacherOutputDataset(path))
                    path = os.path.join(self.args.data[0], '{}_{}_topk_prob'.format(src, tgt))
                    topk_probs.append(TeacherOutputDataset(path))
                    expert_bleu = os.path.join(self.args.data[0], 'expert_bleu_{}_{}.json'.format(src, tgt))
                    expert_bleu = json.load(open(expert_bleu))
                    expert_scores.append(expert_bleu[f"bleu_{src}_{tgt}"])
                return 1

            ds_idx += add_dataset(src, tgt)
            ds_idx += add_dataset(tgt, src)

        src_dataset = ConcatDataset(src_datasets)
        tgt_dataset = ConcatDataset(tgt_datasets)
        src_sizes = np.concatenate([ds.sizes for ds in src_datasets])
        tgt_sizes = np.concatenate([ds.sizes for ds in tgt_datasets])

        topk_idx_dataset = None
        topk_probs_dataset = None
        if is_distill:
            topk_idx_dataset = ConcatDataset(topk_idxs)
            topk_probs_dataset = ConcatDataset(topk_probs)
            assert len(topk_probs_dataset) == len(tgt_dataset), (len(topk_probs_dataset), len(tgt_dataset))
            assert len(topk_idx_dataset) == len(tgt_dataset)

        self.datasets[split] = UniversalDataset(
            self.args,
            src_dataset, src_sizes, self.src_dict, src_lngs, tgt_lngs,
            tgt_dataset, tgt_sizes, self.tgt_dict,
            dataset_ids, lng_borders, dataset_names,
            topk_idxs=topk_idx_dataset, topk_probs=topk_probs_dataset,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            expert_scores=expert_scores,
            is_train=split == 'train'
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

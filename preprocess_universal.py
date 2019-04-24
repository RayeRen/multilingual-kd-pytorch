#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import json
import random
from collections import Counter
from itertools import zip_longest
import os
import shutil

from fairseq.data import indexed_dataset, dictionary
from fairseq.tokenizer import Tokenizer, tokenize_line
from multiprocessing import Pool, Manager, Process


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', metavar='FP', default=None, help='data prefix')
    parser.add_argument('--no-dict', action='store_true', help='do not build dictionary')
    parser.add_argument('--nwordssrc', metavar='N', default=65536, type=int, help='number of target words to retain')
    parser.add_argument('--padding-factor', metavar='N', default=8, type=int,
                        help='Pad dictionary size to be multiple of N')
    parser.add_argument('--joined-dictionary', action='store_true', help='Generate joined dictionary for en-xx')
    parser.add_argument('--expert', default='', type=str)
    parser.add_argument('--workers', metavar='N', default=4, type=int, help='number of parallel workers')
    # parser.add_argument('--workers', metavar='N', default=os.cpu_count(), type=int, help='number of parallel workers')
    return parser


def main(args):
    print(args)
    random.seed(1)

    destdir = os.path.join(args.pref, 'data-bin' + ('' if args.expert == '' else '/' + args.expert))
    os.makedirs(destdir, exist_ok=True)
    dict_path = os.path.join(destdir, 'dict.txt')

    textdir = os.path.join(args.pref, 'text')
    train_dir = os.path.join(textdir, 'train_data')
    test_dir = os.path.join(textdir, 'test_data')
    # if args.expert != '':
    # train_files = glob.glob('{}/train.{}-en.*.e'.format(train_dir, args.expert)) + \
    #               glob.glob('{}/train.en-{}.*.e'.format(train_dir, args.expert))
    # pass
    # else:
    train_files = glob.glob('{}/train.*-*.*'.format(train_dir))
    train_files = [f for f in train_files if len(f.split('.')) in [3, 5]]
    test_files = glob.glob('{}/test.*-*.*'.format(test_dir))
    test_files = [f for f in test_files if len(f.split('.')) in [3, 5]]
    lng_pairs = set([f.split('/')[-1].split(".")[1] for f in (train_files + test_files)])
    print(train_files, test_files, lng_pairs)

    def build_dictionary(filenames):
        d = dictionary.Dictionary()
        for filename in filenames:
            Tokenizer.add_file_to_dictionary(filename, d, tokenize_line, args.workers)
        return d

    tgt_dict_path = os.path.join(destdir, 'dict.tgt.txt')
    if not args.no_dict:
        if args.joined_dictionary:
            src_dict = build_dictionary(train_files)
            src_dict.finalize(
                nwords=args.nwordssrc,
                padding_factor=args.padding_factor
            )
            src_dict.save(dict_path)
            print(src_dict)
        else:
            print("| build en dict.")
            src_dict = build_dictionary([f for f in train_files if f.replace('.tok.bpe', '').endswith('.en')])
            src_dict.finalize(
                nwords=args.nwordssrc,
                padding_factor=args.padding_factor
            )
            src_dict.save(dict_path)

            print("| build xx dict.")
            tgt_dict = build_dictionary([f for f in train_files if not f.replace('.tok.bpe', '').endswith('.en')])
            tgt_dict.finalize(
                nwords=args.nwordssrc,
                padding_factor=args.padding_factor
            )
            tgt_dict.save(tgt_dict_path)

    def make_binary_dataset(input_prefix, output_prefix, lng_pair, lang, num_workers):
        if not args.joined_dictionary and lang != 'en':
            dict = dictionary.Dictionary.load(tgt_dict_path)
        else:
            dict = dictionary.Dictionary.load(dict_path)

        print('| [{}] Dictionary: {} types'.format(lang, len(dict) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result['replaced'])
            n_seq_tok[0] += worker_result['nseq']
            n_seq_tok[1] += worker_result['ntok']

        input_file = f'{input_prefix}.{lng_pair}.{lang}.tok.bpe'
        if not os.path.exists(input_file):
            input_file = f'{input_prefix}.{lng_pair}.{lang}'
            if not os.path.exists(input_file):
                print("| {} not found".format(input_file))
                return
        if args.expert:
            input_file = input_file + '.e'
        offsets = Tokenizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                fn_without_ext = f"{output_prefix}{worker_id}.{lng_pair}.{lang}"
                pool.apply_async(binarize, (input_file, dict, fn_without_ext,
                                            offsets[worker_id],
                                            offsets[worker_id + 1]), callback=merge_result)
            pool.close()

        ds = indexed_dataset.IndexedDatasetBuilder(f"{output_prefix}.{lng_pair}.{lang}.bin")
        merge_result(Tokenizer.binarize(input_file, dict, lambda t: ds.add_item(t),
                                        offset=0, end=offsets[1]))
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                temp_file_path = f"{output_prefix}{worker_id}.{lng_pair}.{lang}"
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(f"{output_prefix}.{lng_pair}.{lang}.idx")

        print('| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}'.format(
            lang, input_file, n_seq_tok[0], n_seq_tok[1],
            100 * sum(replaced.values()) / n_seq_tok[1], dict.unk_word))

    def make_all(lng_pair, lang):
        make_binary_dataset(
            os.path.join(train_dir, 'train'),
            os.path.join(destdir, 'train'),
            lng_pair, lang, num_workers=args.workers)
        make_binary_dataset(
            os.path.join(test_dir, 'test'),
            os.path.join(destdir, 'valid'),
            lng_pair, lang, num_workers=1)

    lngs = set()
    for lng_pair in lng_pairs:
        src_and_tgt = lng_pair.split('-')
        if len(src_and_tgt) != 2:
            continue
        src, tgt = src_and_tgt
        print("| building: ", src, tgt)
        lngs.add(src)
        lngs.add(tgt)
        make_all(lng_pair, src)
        make_all(lng_pair, tgt)

    lngs = list(lngs)
    lngs.sort()
    json.dump(lngs, open(os.path.join(destdir, 'all_lngs.json'), 'w'))


def binarize(filename, dict, fn_without_ext, offset, end):
    ds = indexed_dataset.IndexedDatasetBuilder(f"{fn_without_ext}.bin")

    def consumer(tensor):
        ds.add_item(tensor)

    res = Tokenizer.binarize(filename, dict, consumer, offset=offset, end=end)
    ds.finalize(f"{fn_without_ext}.idx")
    return res


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

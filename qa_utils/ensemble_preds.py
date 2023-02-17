#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import sys
import argparse

import json
import collections


def ensemble_pred(qid, pred_dict_list, vote_score=False, weight=None):
    vote_dict = collections.defaultdict(float)
    for ii, pred_dict in enumerate(pred_dict_list):
        best_ans = pred_dict[qid]["answer"].lower()
        best_ans_score = float(pred_dict[qid]["score"])
        score = best_ans_score if vote_score else 1.0
        if weight:
            score *= weight[ii]
        vote_dict[best_ans] += score

    return sorted(list(vote_dict.items()), key=lambda x:-x[1])[0][0]


def main():
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__
    )
    cmdline_parser.add_argument(
        '--ensemble_list_file',
        help='file contains a list of predictions to be ensembled.',
    )
    cmdline_parser.add_argument(
        "--qid_to_question",
        help="file contains map from qid to question.",
    )
    cmdline_parser.add_argument(
        "--output_file",
        help="output prediction file"
    )
    args = cmdline_parser.parse_args()

    with open(args.ensemble_list_file) as fin:
        file_list = [line.strip().split("\t") for line in fin]

    pred_dict_list = []
    weight_for_pred = []
    for ii, fname in enumerate(file_list):
        with open(fname[0], encoding="utf8") as fin:
            pred_dict_list.append(json.load(fin))

        if len(fname) > 1:
            weight_for_pred.append(float(fname[1]))
        else:
            weight_for_pred.append(1.0)

    qid_list = pred_dict_list[0].keys()

    qid_to_pred = {}

    for qid in qid_list:
        qid_to_pred[qid] = ensemble_pred(qid, pred_dict_list,
                                         weight=weight_for_pred)

    with open(args.output_file, mode="wt", encoding="utf8") as fout:
        fout.write(json.dumps(qid_to_pred, indent=4))


if __name__ == '__main__':
    main()


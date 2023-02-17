#!/usr/bin/env python
"""Prepares for evaluation by merging prediction with original questions."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import sys
import argparse
import json


def main():
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__
    )
    cmdline_parser.add_argument(
        '--pred_file',
        help='prediction json containing qid to predicted answer string.',
    )
    cmdline_parser.add_argument(
        '--qid_to_question_file',
        help='json contains qid to original question.',
    )
    cmdline_parser.add_argument(
        "--output_file",
        help="final output"
    )

    args = cmdline_parser.parse_args()

    with open(args.pred_file, encoding="utf8") as fin:
        qid_to_pred = json.load(fin)

    with open(args.qid_to_question_file, encoding="utf8") as fin:
        qid_to_question = json.load(fin)

    with open(args.output_file, mode="wt", encoding="utf8") as fout:
        for qid, question in qid_to_question.items():
            fout.write(json.dumps({"question": question, "prediction":
                                   qid_to_pred[qid]}))
            fout.write("\n")


if __name__ == '__main__':
    main()


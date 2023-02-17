#!/usr/bin/env python
"""Converts retriever results to DocQA format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import sys
import argparse
import json
import re
import tqdm
from typing import List, Dict, Tuple
import numpy as np


import dateparser


_DATE_FORMAT = {
    # {Month Year}: Janunary 2020.
    "month_year_format": "{date:%B %Y}",
    # {Month Day}: Janunary 1.
    "month_day_format": "{date:%B} {date.day}",
    # {Month Day, Year}: Janunary 1, 2020.
    "month_day_year_format": "{date:%B} {date.day}, {date:%Y}",
}

_SUB_CASE = [
    # Removes all parentheses.
    (r"[\{\}\(\)\[\]]", ""),
    # Removes substrings inside parentheses.
    (r"\([^)]*\)", ""),
    # Removes hypen spaces.
    (r"\s*-\s*", "-"),
    # Removes space before comma.
    (r"\s+,", ","),
    # Removes substrings after double hypen.
    (r"(-\s*-).+", ""),
    # Removes space bfore 's.
    (r"\s+'s", "'s"),
    # Removes space before %.
    (r"\s+%", "%"),
]


def load_gold_context(gold_passage_info_file: str) -> Dict[str, str]:
    """Loads original NQ dataset."""

    with open(gold_passage_info_file, "r", encoding="utf8") as fin:
        data = json.load(fin)["data"]

    # TODO(chenghao): For now, only the canonical questions are used.
    tok_question_to_can_question = dict([
        (sample["question_tokens"], sample["question"])
        for sample in data
        if "question_tokens" in sample and (sample["question"] != sample["question_tokens"])
    ])

    tok_question_to_gold_passage_title = dict([
        (sample["question_tokens"], sample["title"])
        for sample in data
    ])

    return tok_question_to_can_question, tok_question_to_gold_passage_title


def noisy_matching(answer_str: str) -> List[str]:
    """Expands the original answer string into a set of possible mentions."""
    answer_len = len(answer_str.split())
    ans_set = [answer_str] if answer_str else []

    for sub_regx, sub_str in _SUB_CASE:
        new_ans_str = re.sub(sub_regx, sub_str, answer_str).strip()
        if new_ans_str and new_ans_str != answer_str:
            ans_set.append(new_ans_str)

    return ans_set

    if answer_len < 2:
        return ans_set

    # If the answer string is a date, normalizes it using dateparser.
    date = dateparser.parse(answer_str)
    if date:
        year = date.year
        month = date.month
        day = date.day

        month_year_str = _DATE_FORMAT["month_year_format"].format(date=date)
        month_day_str = _DATE_FORMAT["month_day_format"].format(date=date)
        month_day_year_str = _DATE_FORMAT["month_day_year_format"].format(date=date)

        if str(year) in answer_str:
            if answer_len > 2 and answer_str != month_day_year_str:
                ans_set.append(month_day_year_str)
            if answer_str != month_year_str:
                ans_set.append(month_year_str)
        else:
            if answer_str != month_day_str:
                ans_set.append(month_day_str)

        return ans_set

    # Names
    if answer_len > 2:
        if all([word[0].isalpha() and word[0] == word[0].upper()
                for word in answer_str.split()]):
            # Adds an alternative with first and last names only.
            ans_set.append("{0} {1}".format(answer_str.split()[0],
                                            answer_str.split()[-1]))

            return ans_set

    return ans_set


def _get_context_string(context: str, start_idx: int, end_idx: int):
    """Gets the string out of the context."""
    return context[start_idx:end_idx]


def convert_retrieved_passage(
        rank: int, passage: str, answer_list: List[str], question_id: str,
        question_text: str, is_training=True) -> Dict:
    """Annoates retrieved passages with answer spans."""
    # The passage is a Dict with
    # id: String for the passage id.
    # title: String for the passage title.
    # text: String for the passage context.
    # score: Float for the passage ranking score.
    # has_answer: Bool for whether the passage contains the answer string.

    # if is_training:
    #     # has_answer = passage["has_answer"]
    #     # if type(has_answer) is not bool:
    #     #     raise ValueError("has_answer is not of type bool")

    #     passage_text = passage["text"]
    #     passage_id = passage["id"]
    #     passage_doc_title = passage["title"]
    #     passage_rank_score = passage.get("score", 1.0)

    #     answer_span_list = passage["answer_span_list"]
    #     # if passage.get("answer_span_list", False):
    #     #     answer_span_list = passage["answer_span_list"]
    #     # else:
    #     #     answer_span_list = sum(
    #     #         [
    #     #             [
    #     #                 # Finds all mentions of the current answer string.
    #     #                 {"answer_start": item.start(),
    #     #                  "text": _get_context_string(
    #     #                      passage_text, item.start(), item.end())}
    #     #                 for item in re.finditer(re.escape(answer_str), passage_text, re.I)
    #     #             ]
    #     #             # Iterates over all candidate answer string.
    #     #             for answer_str in answer_list
    #     #         ], [])
    # else:
    #     # has_answer = False
    #     passage_text = passage["text"]
    #     passage_id = passage["id"]
    #     passage_doc_title = passage["title"]
    #     passage_rank_score = 1.0
    #     answer_span_list = []

    try:
        passage_text = passage["text"]
    except:
        print(f"No context text for {question_text}")
        raise ValueError(passage)
    passage_id = passage["id"]
    passage_doc_title = passage["title"]
    passage_rank_score = passage.get("score", 1.0)

    answer_span_list = passage["answer_span_list"]

    # answer_span_list = []
    # for answer_str in answer_list:
    #     try:
    #         answer_spans = [
    #             # Finds all mentions of the current answer string.
    #             {"answer_start": item.start(), "text": str(answer_str)}
    #             for item in re.finditer(re.escape(answer_str), passage_text, re.I)
    #         ]

    #         answer_span_list.extend(answer_spans)
    #     except Exception as err:
    #         print(err)
    #         print("Answer string: %s" % answer_str)
    #         print("Passage id: %s" % passage_id)
    #         raise ValueError("Can not parse the answer string!")
    #         # raise ValueError("Can not parse the answer string!")

    # if has_answer and (not answer_span_list):
    #     print("has_answer is True but can not find answer spans passage %s"
    #           % passage_id)
    #     print("question: %s" % question_text)

    # elif (not has_answer) and answer_span_list:
    #     print("has_answer is False but can find answer spans passage %s"
    #           % passage_id)
    #     print("question: %s" % question_text)
    #     print(answer_list)

    return {
        "context": passage_text,
        "doc_title": passage_doc_title,
        "passage_id": passage_id,
        "passage_rank_score": passage_rank_score,
        "has_answer": bool(answer_span_list),
        "qas": [{
            "id": "{0}-{1}".format(question_id, rank),
            "is_impossible": not bool(answer_span_list),
            "qid": question_id,
            "question": question_text,
            "answers": answer_span_list,
        }]
    }


def _is_from_gold_page(gold_info, psg_title, question):
    """Checks whether the given passage is from the gold Wiki page."""
    gold_page_title = gold_info.get(question, None)
    if gold_page_title:
        return (psg_title == gold_page_title)
    return False


def convert_single_question_with_context_passage(
        question_id, question_text, answer_list, passage_list, is_training=False,
        gold_info=None, tok_question=None, max_positives=20, max_negatives=60,
        min_negatives=150, do_noise_match=True, use_heuristic=True,
        answer_match_w_regex=False):
    """Converts a single question with retrieved passages to reader examples."""
    if not answer_match_w_regex:
        if do_noise_match:
            expand_answer_list = list(set(sum([noisy_matching(answer_str)
                                      for answer_str in answer_list], [])))
        else:
            expand_answer_list = list(set([ans.lower() for ans in answer_list]))
    else:
        expand_answer_list = answer_list

    neg_doc = True

    # positive_passages, negative_passages = [], []
    annotate_psgs = []
    for psg in passage_list:
        new_psg = dict(psg)
        passage_text = psg["text"]
        if answer_match_w_regex:
            try:
                answer_span_list = sum(
                    [
                        [
                            # Finds all mentions of the current answer string.
                            {"answer_start": item.start(),
                             "text": _get_context_string(
                                 passage_text, item.start(), item.end())}
                            for item in re.finditer("(%s)" % answer_str, passage_text, re.I)
                        ]
                        # Iterates over all candidate answer string.
                        for answer_str in expand_answer_list
                    ], [])
            except:
                print("question: %s" % question_text)
                answer_span_list = []
        else:
            answer_span_list = sum(
                [
                    [
                        # Finds all mentions of the current answer string.
                        {"answer_start": item.start(),
                         "text": _get_context_string(
                             passage_text, item.start(), item.end())}
                        for item in re.finditer(re.escape(answer_str), passage_text, re.I)
                    ]
                    # Iterates over all candidate answer string.
                    for answer_str in expand_answer_list
                ], [])


        if answer_span_list:
            new_psg["answer_span_list"] = answer_span_list
            new_psg["has_answer"] = True
            neg_doc = False
        else:
            new_psg["answer_span_list"] = []
            new_psg["has_answer"] = False

        annotate_psgs.append(new_psg)

    if (not is_training) or (not use_heuristic):
        return {
            "paragraphs": [
                convert_retrieved_passage(
                    rank, context_passage, expand_answer_list, question_id, question_text)
                for rank, context_passage in enumerate(annotate_psgs)
            ],
            "question": question_text,
            "question_id": question_id,
            "origin_answer_strings": answer_list,
            "expand_answer_strings": expand_answer_list,
            "neg_doc": neg_doc,
        }

    positive_passages, negative_passages = [], []
    for psg in annotate_psgs:
        if psg["has_answer"]:
            positive_passages.append(psg)
        else:
            negative_passages.append(psg)

    positive_passages = sorted(
        positive_passages, key=lambda psg: -float(psg.get("score", 1.0)))

    negative_passages = sorted(
        negative_passages, key=lambda psg: -float(psg.get("score", 1.0)))

    if gold_info and tok_question is None:
        raise ValueError(
            "If gold_info is given, tok_question can not be None")

    if not negative_passages:
        print(f"There is no negative for {question_id}, {question_text}")

    selected_positive_psgs = []
    if gold_info:
        # Processes the passage_list with the given gold passage information.
        positive_psg_from_gold_page = sorted(
            list(filter(lambda psg: _is_from_gold_page(
                gold_info, psg["title"], tok_question), positive_passages)),
            key=lambda psg: -float(psg.get("score", 1.0)))

        selected_positive_psgs = list(
            filter(
                lambda psg: psg["has_answer"],
                [convert_retrieved_passage(rank, psg, expand_answer_list,
                                           question_id, question_text)
                 for rank, psg in enumerate(positive_psg_from_gold_page)]))

    possible_negatives = []
    if not selected_positive_psgs:
        # Falls back to use all positive passages.
        all_ranked_positives = list(filter(
            lambda psg: psg["has_answer"],
            [convert_retrieved_passage(
                rank, psg, expand_answer_list, question_id, question_text)
             for rank, psg in enumerate(positive_passages)]))
        selected_positive_psgs = all_ranked_positives[0:max_positives]
        # TODO(chehao): Fix this.
        possible_negatives = all_ranked_positives[max_positives:]

    # if not negative_passages:
    #     # If gold info is provided and no negative, considers those as negative.
    #     if gold_info:
    #         negative_passages = [
    #             psg for psg in positive_passages if not _is_from_gold_page(
    #                 gold_info, psg["title"], tok_question)
    #         ]
    #     else:
    #         negative_passages = possible_negatives

    num_positives = len(selected_positive_psgs)
    max_negs = min(max(10 * num_positives, max_negatives), min_negatives)

    neg_converted_psgs = [
        convert_retrieved_passage(
            num_positives + rank, psg, [], question_id, question_text)
        for rank, psg in enumerate(negative_passages[0:max_negs])
    ]

    return {
        "paragraphs": selected_positive_psgs + neg_converted_psgs,
        "question": question_text,
        "question_id": question_id,
        "origin_answer_strings": answer_list,
        "expand_answer_strings": expand_answer_list,
        "neg_doc": neg_doc,
        "num_positives": num_positives,
        "num_negatives": len(neg_converted_psgs),
    }


def convert_retrieval_result(input_file, output_file, offset=100000,
                             dataset="nq", qid_qas_file=None,
                             do_noise_match=False,
                             prefix="nq", gold_qp_info_file=None,
                             is_training=False, use_heuristic=False):
    """Converts retriever results to DocQA format."""
    print("Converting %s" % input_file)
    with open(input_file, "r", encoding="utf8") as fin:
        input_data = json.load(fin)

    if gold_qp_info_file:
        print("Gold passage question information is provided")
        tok_q_to_cano_q, tok_q_to_gold_title = load_gold_context(gold_qp_info_file)
        print(
            "There are %d tokenized question are different from canonical ones"
            % len(tok_q_to_cano_q))

        if is_training:
            print("keeps gold passage info")
        else:
            print("throws out gold passage info for non-training splits")
            tok_q_to_gold_title = None

    else:
        print("There is no gold passage question information provided")
        tok_q_to_cano_q = {}
        tok_q_to_gold_title = None

    print("Number of examples to be processed %d" % len(input_data))

    # TODO(chenghao): where can we find the question id?
    # Retriever result json format:
    # A list of examples where each example is a Dict in the form of
    # "question": String for the question which is not the canonical form.
    # "answers": List of answer strings.
    # "ctxs": List of passage, where each passage is a Dict in the form of
    #         "id": String for the passage id.
    #         "title": String for the document title.
    #         "score": Float for the retrieval score.
    #         "has_answer": Bool for whether the passage contains an answer.

    # data = []
    # for ii, sample in tqdm.tqdm(enumerate(input_data)):
    #     if ii > 53: break
    #     data.append(
    #         convert_single_question_with_context_passage(
    #             "{0}_{1}".format(prefix, ii + offset),
    #             sample["question"], sample["answers"], sample["ctxs"]))

    # output_data = {
    #     "data": data,
    #     "version": "v2.0",
    # }

    # do_noise_match = False
    if dataset == "nq":
        print("Typically, noisy matching is used for NQ")
        if not do_noise_match:
            print("do_noise_match is turned off")

    answer_match_w_regex = False
    if dataset == "trec":
        print("Carries out regexp match for CuratedTrec")
        answer_match_w_regex = True

    output_data = {
        "data": [
            convert_single_question_with_context_passage(
                "{0}_{1}".format(prefix, ii + offset),
                tok_q_to_cano_q.get(sample["question"], sample["question"]),
                sample["answers"] if "answers" in sample else [], sample["ctxs"],
                gold_info=tok_q_to_gold_title,
                tok_question=sample["question"],
                is_training=is_training,
                do_noise_match=do_noise_match,
                answer_match_w_regex=answer_match_w_regex,
                use_heuristic=use_heuristic,
            )
            for ii, sample in tqdm.tqdm(enumerate(input_data))
        ],
        "version": "2.0"
    }

    if qid_qas_file:
        qid_to_qas = dict([
            (
                datum["question_id"],
                {
                    "question": datum["question"],
                    "answers": datum["origin_answer_strings"],
                }
            )
            for datum in output_data["data"]
        ])
        with open(qid_qas_file, mode="wt") as fout:
            json.dump(qid_to_qas, fout, indent=4)

    print("Number of processed examples %d" % len(output_data["data"]))
    pos_doc_count = 0
    ctx_sizes = []
    pos_neg_ratios = []
    for dd in output_data["data"]:
        pos_cnt = 0
        if not dd["neg_doc"]:
            pos_doc_count += 1
        for pp in dd["paragraphs"]:
            if pp["has_answer"]:
                pos_cnt += 1

        ctx_sizes.append(len(dd["paragraphs"]))
        pos_neg_ratios.append(float(pos_cnt) / len(dd["paragraphs"]))

    if pos_doc_count == 0:
        if is_training:
            raise ValueError("There is no positive document!")
        print(f"There is no positive document for {input_file}!")

    print("Retrieval Accuracy %f" % (
        float(pos_doc_count) / len(output_data["data"])))


    print("75 percentile num of contexts", np.percentile(ctx_sizes, 75))
    print("min num of contexts", np.min(ctx_sizes))
    print("mean num of contexts", np.mean(ctx_sizes))
    print("75 percentile pos neg ratio", np.percentile(pos_neg_ratios, 75))
    print("min pos neg ratio", np.min(pos_neg_ratios))
    print("mean pos neg ratio", np.mean(pos_neg_ratios))
    print("Writing converted file to %s" % output_file)
    with open(output_file, mode='wt', encoding="utf8") as fout:
        fout.write(json.dumps(output_data))


def main():
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__
    )
    cmdline_parser.add_argument(
        '--input_dir',
        type=str,
        help='input retrieval result directory',
    )
    cmdline_parser.add_argument(
        '--output_dir',
        type=str,
        help='output directory',
    )

    cmdline_parser.add_argument(
        '--splits',
        type=str,
        default=[],
        help='split to be processed, sep by `,`',
    )
    cmdline_parser.add_argument(
        '--retriever_result_ext',
        type=str,
        default="json",
        help='retriever result file extension',
    )
    cmdline_parser.add_argument(
        '--gold_passage_info_dir',
        type=str,
        help='gold passage directory with canonical passages and questions',
    )
    cmdline_parser.add_argument(
        '--dataset',
        type=str,
        default="nq",
        choices=["nq", "trivia", "wq", "trec", "wikisum", "wikitable", "ott"],
        help='dataset for processing',
    )

    cmdline_parser.add_argument(
        '--has_ans_annotate',
        type=str,
        default="true",
        choices=["true", "false"],
        help='wether to include answer annotation for conversion',
    )

    cmdline_parser.add_argument(
        '--use_heuristic',
        type=str,
        default="true",
        choices=["true", "false"],
        help='whether to use heuristic for conversion',
    )
    cmdline_parser.add_argument(
        '--do_noise_match',
        type=str,
        default="false",
        choices=["true", "false"],
        help='whether to heuristics for answer matching',
    )


    args = cmdline_parser.parse_args()
    use_heuristic = (args.use_heuristic == 'true')
    do_noise_match = (args.do_noise_match == 'true')

    for split in args.splits.split(","):
        input_file = os.path.join(
            args.input_dir, "{0}.{1}".format(split, args.retriever_result_ext))

        output_file = os.path.join(
            args.output_dir, "{0}{1}".format(split, "-v2.0.json"))

        is_training = ("train" in split)

        gold_passage_info_file = None
        if args.gold_passage_info_dir:
            # Only NQ has the gold info.
            gold_passage_info_file = os.path.join(
                args.gold_passage_info_dir, "nq_{0}.json".format(split))
            if split == "merged_train":
                gold_passage_info_file = os.path.join(
                    args.gold_passage_info_dir, "{0}_gold_info.json".format(split))


        if not gold_passage_info_file and args.dataset == "nq":
            raise ValueError("gold_passage_info_dir is not provided!")

        qid_qas_file = None
        if not is_training:
            qid_qas_file = os.path.join(
                args.output_dir, f"nq_single_qid.{split}.json")
            print(f"Outputs qid_qas file to {qid_qas_file}")

        convert_retrieval_result(input_file, output_file,
                                 gold_qp_info_file=gold_passage_info_file,
                                 prefix=args.dataset,
                                 dataset=args.dataset,
                                 qid_qas_file=qid_qas_file,
                                 is_training=is_training,
                                 do_noise_match=do_noise_match,
                                 use_heuristic=use_heuristic)


if __name__ == '__main__':
    main()


#!/usr/bin/env python
"""Converts QA examples into TFrecord."""


from __future__ import absolute_import, division, print_function

import os
import random

import tensorflow as tf
import tokenization
from transformers import RobertaTokenizer
from utils.create_doc_qa_data import (
    read_doc_squad_examples_from_generator,
    DocFeatureJsonWriter,
    convert_doc_examples_to_doc_features,
)


flags = tf.flags

FLAGS = flags.FLAGS

# Data parameters
flags.DEFINE_string("json_file", None,
                    "SQuAD json for conversion. E.g., train-v1.1.json")

flags.DEFINE_string("split_name", None,
                    "The split prefix name for output file, such as train.")

flags.DEFINE_integer("num_split", 8, "the number of splits")
flags.DEFINE_integer("split", None, "split id in [0, num_split].")

flags.DEFINE_bool("is_training", True,
                  "Whether the processed file is used for training.")

flags.DEFINE_bool("pad_example", True,
                  "Whether to pad irrelevant examples.")

flags.DEFINE_integer("keep_topk", 50,
                     "Max number of passages for processing.")

flags.DEFINE_string("output_dir", None, "The output directory.")


# Data processing specific parameters.
# The sequence length is very important.
flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("use_doc_title", True,
                  "Whether to include document title as part of query.")

# Document-level QA parameters.
flags.DEFINE_integer("max_num_doc_feature", 12,
                     "Max number of document features allowed.")

flags.DEFINE_integer("max_short_answers", 10,
                     "The maximum number of distinct short answer positions.")

flags.DEFINE_integer("max_num_dupe_factor", 6,
                     "The maximum number of dupe samples.")


flags.DEFINE_integer("max_num_answer_strings", 80,
                     "The maximum number of distinct short answer strings.")

flags.DEFINE_string("no_answer_string", "",
                    "The string is used for as no-answer string.")

flags.DEFINE_bool("filter_null_doc", True,
                  "Whether to filter out no-answer document.")

flags.DEFINE_bool("single_pos_per_dupe", True,
                  "Whether to sample only one positive per duplicate.")

flags.DEFINE_bool("debug", False, "If true we process a tiny dataset.")

flags.DEFINE_bool(
    "version_2_with_negative", True,
    "If true, the SQuAD examples contain some that do not have an answer.")


# Tokenizer parameters.
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("merges_file", None,
                    "The vocabulary file that the RoBERTa model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string(
    "model_type", "bert",
    "Model type will determine the corresponding tokenization.")

flags.DEFINE_integer("rand_seed", 12345, "The random seed used")


def validate_flags_or_throw():
  """Validate the input FLAGS or throw an exception."""
  if FLAGS.max_seq_length > 512:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, 512))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))

  if FLAGS.max_num_answer_strings < FLAGS.max_short_answers:
      raise ValueError(
          "The max_num_answer_strings (%d) must be bigger than "
          "max_short_answers (%d)" % (FLAGS.max_num_answer_strings,
                                      FLAGS.max_short_answers)
      )
  if FLAGS.model_type == "roberta":
      if FLAGS.merges_file is None:
        raise ValueError(
            "When using Roberta tokenizer, merges_file must be defined!")


def main(_):
    """Main function."""
    validate_flags_or_throw()

    if FLAGS.model_type == "bert":
        # Loads BERT tokenizer.
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    elif FLAGS.model_type == "roberta":
        tokenizer = RobertaTokenizer(FLAGS.vocab_file, FLAGS.merges_file)
    else:
        raise ValueError("Unknown model type %s" % FLAGS.model_type)

    train_examples, _ = read_doc_squad_examples_from_generator(
        input_file=FLAGS.json_file, is_training=FLAGS.is_training,
        keep_topk=FLAGS.keep_topk, pad_example=FLAGS.pad_example
    )
    tf.logging.info("Total number of document exmples %d" % len(train_examples))

    if FLAGS.use_doc_title:
        if train_examples[0].example_list[0].title is None:
            raise ValueError(
                "When use_doc_tile=True, doc_title is required in the input!")
        tf.logging.info("Inlcudes document tile for conversion.")

    start_index = 0
    end_index = len(train_examples)
    num_examples_per_split = len(train_examples) // FLAGS.num_split
    remainder = len(train_examples) % FLAGS.num_split

    if FLAGS.split < remainder:
        start_index = FLAGS.split * (num_examples_per_split + 1)
        end_index = start_index + num_examples_per_split + 1
    else:
        start_index = FLAGS.split * num_examples_per_split + remainder
        end_index = start_index + num_examples_per_split

    if FLAGS.single_pos_per_dupe:
        tf.logging.info("Keeps only one positive for each duplicate.")

    doc_examples=train_examples[start_index:end_index]
    output_filename = os.path.join(
        FLAGS.output_dir, "{0}.tf_record_{1}".format(
            FLAGS.split_name, FLAGS.split))

    train_writer = DocFeatureJsonWriter(
        filename=output_filename,
    )

    convert_doc_examples_to_doc_features(
        doc_examples=doc_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=FLAGS.is_training,
        output_fn=train_writer.process_doc_feature,
        include_title=FLAGS.use_doc_title,
        model_type=FLAGS.model_type,
    )
    train_writer.close()

    num_split_examples = train_writer.num_features
    tf.logging.info("  Num orig examples = %d", end_index - start_index)
    tf.logging.info("  Num split examples = %d", num_split_examples)


if __name__ == '__main__':
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("json_file")
    tf.app.run()

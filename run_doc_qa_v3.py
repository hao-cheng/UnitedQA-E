# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import, division, print_function

import collections
import json
import math
import os
import random
import shutil
import time
import re

import horovod.tensorflow as hvd
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.client import device_lib

import modeling_relpo as modeling_func
import optimization
import tokenization
from transformers import RobertaTokenizer
from utils.create_doc_qa_data import (
    read_doc_squad_examples_from_generator,
    read_squad_examples_from_generator,
    DocFeatureWriter,
    FeatureWriter,
    convert_doc_examples_to_doc_features,
    convert_examples_to_features,
    bert_doc_token_processor,
    roberta_doc_token_processor,
    compute_doc_norm_score,
    InputFeatures,
)
from utils.utils import LogEvalRunHook, LogTrainRunHook
from qa_utils.doc_qa_loss_helper import answer_pos_loss

from qa_utils.loss_helper import (
    compute_double_forward_loss_w_add_noise_correct_only,
    compute_double_forward_loss_w_add_noise,
    compute_double_forward_loss_w_add_noise_span,
    compute_vat_loss,
    compute_approx_jacobian_loss,
    get_embeddings,
    compute_forward_logits,
)


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("merges_file", None,
                    "The vocabulary file that the RoBERTa model was trained on.")


flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "tfrecord_dir", None,
    "Directory with input files, comma separated or single directory.")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string(
    "model_type", "bert",
    "Model type will determine the corresponding tokenization.")

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

flags.DEFINE_integer(
    "infer_forward_k", 0,
    "The number of inference time noisy forward")

flags.DEFINE_integer("rand_seed", 12345, "The random seed used")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("num_eval_split", 4, "The number of parallel evaluations.")
flags.DEFINE_integer("eval_split_id", 4, "The index of the recurrent evaluation.")

flags.DEFINE_bool("use_doc_title", False,
                  "Whether to include document title as part of query.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_integer("max_num_doc_feature", 12,
                     "Max number of document features allowed.")

flags.DEFINE_integer("topk_for_infer", 50,
                     "Max number of passages for inference.")

flags.DEFINE_float("learning_rate", 5e-6, "The initial learning rate for Adam.")

flags.DEFINE_bool("use_trt", False, "Whether to use TF-TRT")

flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")
# flags.DEFINE_float("num_train_epochs", 3.0,
#                    "Total number of training epochs to perform.")

flags.DEFINE_integer("num_train_steps", 0,
                     "Total number of training steps to perform.")


flags.DEFINE_float(
    "warmup_proportion", 0.05,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("num_accumulation_steps", 1,
                     "Number of accumulation steps before gradient update"
                      "Global batch size = num_accumulation_steps * train_batch_size")

flags.DEFINE_string(
    "optimizer_type", "adam",
    "Optimizer used for training - LAMB or ADAM")

flags.DEFINE_integer(
    "n_best_size", 10,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")


flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

# Document-level QA parameters.
flags.DEFINE_integer("max_short_answers", 10,
                     "The maximum number of distinct short answer positions.")

flags.DEFINE_integer("max_num_answer_strings", 80,
                     "The maximum number of distinct short answer strings.")

flags.DEFINE_string("no_answer_string", "",
                    "The string is used for as no-answer string.")

flags.DEFINE_bool("filter_null_doc", True,
                  "Whether to filter out no-answer document.")

flags.DEFINE_bool("debug", False, "If true we process a tiny dataset.")

flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")
flags.DEFINE_integer("num_eval_iterations", None,
                     "How many eval iterations to run - performs inference on subset")

# TRTIS Specific flags
flags.DEFINE_bool("export_trtis", False, "Whether to export saved model or run inference with TRTIS")
flags.DEFINE_string("trtis_model_name", "bert", "exports to appropriate directory for TRTIS")
flags.DEFINE_integer("trtis_model_version", 1, "exports to appropriate directory for TRTIS")
flags.DEFINE_string("trtis_server_url", "localhost:8001", "exports to appropriate directory for TRTIS")
flags.DEFINE_bool("trtis_model_overwrite", False, "If True, will overwrite an existing directory with the specified 'model_name' and 'version_name'")
flags.DEFINE_integer("trtis_max_batch_size", 8, "Specifies the 'max_batch_size' in the TRTIS model config. See the TRTIS documentation for more info.")
flags.DEFINE_float("trtis_dyn_batching_delay", 0, "Determines the dynamic_batching queue delay in milliseconds(ms) for the TRTIS model config. Use '0' or '-1' to specify static batching. See the TRTIS documentation for more info.")
flags.DEFINE_integer("trtis_engine_count", 1, "Specifies the 'instance_group' count value in the TRTIS model config. See the TRTIS documentation for more info.")

# Robustness-related flags.
flags.DEFINE_float(
    "double_forward_reg_rate", 0.0,
    "Weight for controlling the double forward loss."
)

flags.DEFINE_string(
    "double_forward_loss", "v2",
    "Double forward loss type, {v1, v2, v3}"
)

flags.DEFINE_float(
    "noise_epsilon", 1e-3,
    "Float for the random noise scale"
)

flags.DEFINE_string(
    "noise_normalizer", "L2",
    "String for the noise normalizer type {L2, L1, Linf}"
)

flags.DEFINE_float(
    "kl_alpha", 1e-3,
    "Float for scaling the entropy term of KL"
)

flags.DEFINE_float(
    "kl_beta", 1.0,
    "Float for scaling the cross entropy term of KL"
)

flags.DEFINE_float(
    "jacobian_reg_rate", 0.0,
    "Weight for controlling the jacobian loss.")

flags.DEFINE_float(
    "vat_reg_rate", 0.0,
    "Weight for controlling the vat loss.")

# Multi-passage losses.
flags.DEFINE_string("global_loss", "doc_pos-mml",
                    "The global multi-passage objective")

# Single-passage losses.
flags.DEFINE_string("local_loss", "None",
                    "The local passage-level objective")


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  if bert_config.electra:
      scope = "electra"
  else:
      scope = "bert"

  model = modeling_func.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      compute_type=tf.float32,
      scope=scope
  )

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling_func.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0, name='unstack')

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  # Adds relevance prediction on top of pooled_output with shape
  # [batch_size, hidden_size].
  pooled_output = model.get_pooled_output()
  relevance_weights = tf.get_variable(
      "cls/squad/relevance_weights", [hidden_size, 1],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  relevance_bias = tf.get_variable(
      "cls/squad/relevance_bias", [1], initializer=tf.zeros_initializer())

  relevance_logits = tf.matmul(pooled_output, relevance_weights)
  relevance_logits = tf.nn.bias_add(relevance_logits, relevance_bias)

  return (start_logits, end_logits, relevance_logits, model)


def reshape_doc_feature(orig_tensor, batch_size, depth, num_features):
    """Reshapes input from [batch_size, num_features * depth] to
        [batch_size * num_features, depth].
    """
    return tf.reshape(orig_tensor, [batch_size * num_features, depth])


def flat_doc_feature(doc_tensor, batch_size, depth, num_features):
    """Reshapes input from [batch_size * num_features, depth] to
        [batch_size, num_features * depth].
    """
    return tf.reshape(doc_tensor, [batch_size, num_features * depth])


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     hvd=None, use_fp16=False, use_one_hot_embeddings=False,
                     num_doc_features=1, max_short_answers=1,
                     seq_length=FLAGS.max_seq_length):
  """Returns `model_fn` closure for Estimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""
    if FLAGS.verbose_logging:
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if is_training:
        # The inputs are of shape [batch_size, seq_length * num_doc_features].
        input_shape = modeling_func.get_shape_list(input_ids, expected_rank=2)

        batch_size = input_shape[0]

        input_ids = reshape_doc_feature(input_ids, batch_size, seq_length, num_doc_features)
        input_mask = reshape_doc_feature(input_mask, batch_size, seq_length, num_doc_features)
        segment_ids = reshape_doc_feature(segment_ids, batch_size, seq_length, num_doc_features)

    (start_logits, end_logits, relevance_logits, model) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )

    if bert_config.use_rel_pos_embeddings:
      tf.logging.info("***Uses relative position embeddings***")

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    if init_checkpoint and (hvd is None or hvd.rank() == 0):
      (assignment_map, initialized_variable_names
      ) = modeling_func.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if FLAGS.verbose_logging:
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          tf.logging.info(" %d name = %s, shape = %s%s", 0 if hvd is None else hvd.rank(), var.name, var.shape,
                          init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      start_positions = features["start_positions"]
      end_positions = features["end_positions"]
      # answer_positions_mask = features["answer_positions_mask"]

      start_positions = reshape_doc_feature(start_positions, batch_size, max_short_answers, num_doc_features)
      end_positions = reshape_doc_feature(end_positions, batch_size, max_short_answers, num_doc_features)
      # answer_positions_mask = reshape_doc_feature(answer_positions_mask, batch_size, max_short_answers, num_doc_features)

      start_logits = tf.cast(start_logits, tf.float32)
      end_logits = tf.cast(end_logits, tf.float32)

      total_loss = 0.0

      # Prepares for multi-passage losses.
      if FLAGS.global_loss == "doc_pos-mml":
        tf.logging.info("Using document-level normalization MML loss!")
        doc_feature_size = seq_length * num_doc_features
        position_offset = tf.tile(tf.reshape(
            tf.range(0, doc_feature_size, seq_length, dtype=tf.int32),
            [1, num_doc_features]), [batch_size, 1])
        position_offset = reshape_doc_feature(
            position_offset, batch_size, 1, num_doc_features)

        flat_start_logits = flat_doc_feature(
            start_logits, batch_size, seq_length, num_doc_features)
        flat_end_logits = flat_doc_feature(
            end_logits, batch_size, seq_length, num_doc_features)

        flat_start_positions = flat_doc_feature(
          start_positions + position_offset, batch_size, max_short_answers, num_doc_features)
        flat_end_positions = flat_doc_feature(
          end_positions + position_offset, batch_size, max_short_answers, num_doc_features)
        flat_seq_length = seq_length * num_doc_features

        flat_answer_positions_mask = features["answer_positions_mask"] * features["notnull_answer_mask"]
        total_loss += answer_pos_loss(
            start_logits=flat_start_logits,
            start_indices=flat_start_positions,
            end_logits=flat_end_logits,
            end_indices=flat_end_positions,
            answer_positions_mask=flat_answer_positions_mask,
            seq_length=flat_seq_length,
            loss_type="par_mml",
        )

      if FLAGS.local_loss == "par_pos-mml":
        tf.logging.info("Using passage-level normalization MML loss!")
        flat_start_logits = start_logits
        flat_end_logits = end_logits

        flat_start_positions = start_positions
        flat_end_positions = end_positions
        flat_seq_length = seq_length
        flat_answer_positions_mask = reshape_doc_feature(
            features["answer_positions_mask"], batch_size, max_short_answers, num_doc_features)

        total_loss += answer_pos_loss(
            start_logits=flat_start_logits,
            start_indices=flat_start_positions,
            end_logits=flat_end_logits,
            end_indices=flat_end_positions,
            answer_positions_mask=flat_answer_positions_mask,
            seq_length=flat_seq_length,
            loss_type="par_mml",
        )

      clean_loss = total_loss

      if FLAGS.vat_reg_rate > 0.0:
          vat_loss = compute_vat_loss(start_logits, end_logits, model)
          total_loss += FLAGS.vat_reg_rate * vat_loss
      else:
          vat_loss = tf.constant(0.0)

      if FLAGS.jacobian_reg_rate > 0.0:
          jacobian_loss = compute_approx_jacobian_loss(
              start_logits, end_logits, model.get_embedding_output())
          total_loss += FLAGS.jacobian_reg_rate * jacobian_loss
      else:
          jacobian_loss = tf.constant(0.0)

      if FLAGS.double_forward_reg_rate > 0.0:
        k = 20
        double_forward_loss = compute_double_forward_loss_w_add_noise_span(
            start_logits, end_logits, model,
            loss_type=FLAGS.double_forward_loss,
            noise_epsilon=FLAGS.noise_epsilon,
            noise_normalizer=FLAGS.noise_normalizer,
            alpha=FLAGS.kl_alpha,
            beta=FLAGS.kl_beta,
            k=k,
        )
        total_loss += FLAGS.double_forward_reg_rate * double_forward_loss
      else:
        double_forward_loss = tf.constant(0.0)

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, hvd=hvd,
          manual_fp16=False, use_fp16=use_fp16,
          num_accumulation_steps=FLAGS.num_accumulation_steps,
          optimizer_type=FLAGS.optimizer_type,
      )

      logging_hook = tf.train.LoggingTensorHook(
          {"total_loss": total_loss, "clean_loss": clean_loss,
           "jacobian_loss": jacobian_loss,
           "scaled_jacobian_loss": FLAGS.jacobian_reg_rate * jacobian_loss,
           "vat_loss": vat_loss,
           "scaled_vat_loss": FLAGS.vat_reg_rate * vat_loss,
           "double_forward_loss": double_forward_loss,
           "scaled_double_forward_loss": FLAGS.double_forward_reg_rate * double_forward_loss,
           },
          every_n_iter=100)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          training_hooks=[logging_hook],
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      k = FLAGS.infer_forward_k
      if k > 0:
        tf.logging.info("Carries out %d inference noisy forward" % k)
        for _ in range(k):
          (d_start_logits, d_end_logits) = compute_forward_logits(model.adv_forward(
            get_embeddings(model, "L2", 1e-3))[0])
          start_logits += d_start_logits
          end_logits += d_end_logits
        start_logits -= tf.log(float(k + 1))
        end_logits -= tf.log(float(k + 1))
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
          "relevance_logits": relevance_logits,
      }
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode, predictions=predictions)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def doc_input_fn_builder(input_files, batch_size, seq_length, is_training,
                         drop_remainder, hvd=None, num_feature_per_doc=1,
                         max_num_answers=1, num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to Estimator."""

  if not is_training:
      raise ValueError("The doc_input_fn_builder only supports training")

  input_length = seq_length * num_feature_per_doc
  label_length = max_num_answers * num_feature_per_doc
  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([input_length], tf.int64),
      "input_mask": tf.FixedLenFeature([input_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([input_length], tf.int64),
      "start_positions": tf.FixedLenFeature([label_length], tf.int64),
      "end_positions": tf.FixedLenFeature([label_length], tf.int64),
      "answer_positions_mask": tf.FixedLenFeature([label_length], tf.int64),
      "answer_ids": tf.FixedLenFeature([label_length], tf.int64),
      "notnull_answer_mask": tf.FixedLenFeature([label_length], tf.int64),
  }

  def input_fn():
    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files))

    # `cycle_length` is the number of parallel files that get read.
    cycle_length = min(num_cpu_threads, len(input_files))

    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    d = d.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=is_training,
            cycle_length=cycle_length))
    d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True if is_training else False))

    return d

  return input_fn


def input_fn_builder(input_file, batch_size, seq_length, is_training, drop_remainder, hvd=None):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn():
    """The actual input function."""

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        d = tf.data.TFRecordDataset(input_file, num_parallel_reads=4)
        if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
        d = d.apply(tf.data.experimental.ignore_errors())
        d = d.shuffle(buffer_size=100)
        d = d.repeat()
    else:
        d = tf.data.TFRecordDataset(input_file)


    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn



RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits",
                                    "relevance_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      prob_transform_func, unique_id_to_doc_score,
                      model_type="bert", tokenizer=None):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index",
         "start_logit", "end_logit", "doc_score", "relevance_score"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        null_doc_score = 0
        null_relevance_score = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if FLAGS.version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
                    null_doc_score = unique_id_to_doc_score[result.unique_id]
                    null_relevance_score = result.relevance_logits
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            relevance_score=result.relevance_logits,
                            doc_score=unique_id_to_doc_score[result.unique_id],
                        ))

        if FLAGS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    relevance_score=null_relevance_score,
                    doc_score=null_doc_score))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction",
            ["text", "start_logit", "end_logit", "doc_score", "relevance_score"])

        ner_tag_regx = re.compile("(\s?S0\S+\s)|(\sE0\S+\s?)", re.I)
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]

                if model_type == "bert":
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")
                else:
                    # Converts back to normal string.
                    tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Uses NER tag token patterns to strip out tags.
                tok_text = re.sub(ner_tag_regx, " ", tok_text)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                # Uses NER tag token patterns to strip out tags.
                orig_text = re.sub(ner_tag_regx, " ", orig_text)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    relevance_score=pred.relevance_score,
                    doc_score=pred.doc_score))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if FLAGS.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        relevance_score=null_relevance_score,
                        doc_score=null_doc_score))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="", start_logit=0.0, end_logit=0.0, doc_score=0.0,
                    relevance_score=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = prob_transform_func(total_scores)

        if not best_non_null_entry:
            tf.logging.info("No non-null guess")
            best_non_null_entry = _NbestPrediction(
                text="", start_logit=0.0, end_logit=0.0, doc_score=0.0,
                relevance_score=0.0)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["relevance_score"] = entry.relevance_score
            output["doc_score"] = entry.doc_score
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not FLAGS.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > FLAGS.null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    tf.logging.info("Dumps predictions")
    with tf.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if FLAGS.version_2_with_negative:
        with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  # When we created the data, we kept track of the alignment between original
  # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
  # now `orig_text` contains the span of our original text corresponding to the
  # span that we predicted.
  #
  # However, `orig_text` may contain extra characters that we don't want in
  # our prediction.
  #
  # For example, let's say:
  #   pred_text = steve smith
  #   orig_text = Steve Smith's
  #
  # We don't want to return `orig_text` because it contains the extra "'s".
  #
  # We don't want to return `pred_text` because it's already been normalized
  # (the SQuAD eval script also does punctuation stripping/lower casing but
  # our tokenizer does additional normalization like stripping accent
  # characters).
  #
  # What we really want to return is "Steve Smith".
  #
  # Therefore, we have to apply a semi-complicated alignment heruistic between
  # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
  # can fail in certain cases in which case we just return `orig_text`.

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs



def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict and not FLAGS.export_trtis:
    raise ValueError("At least one of `do_train` or `do_predict` or `export_SavedModel` must be True.")

  if FLAGS.do_train:
    if not FLAGS.tfrecord_dir:
      raise ValueError(
          "If `do_train` is True, then `tfrecord_dir` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

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

  if bert_config.roberta:
      if FLAGS.merges_file is None:
        raise ValueError(
            "When using Roberta tokenizer, merges_file must be defined!")

  # if FLAGS.local_obj_alpha > 0.0:
  #     tf.logging.info("Using local_obj_alpha=%f" % FLAGS.local_obj_alpha)


def prediction_generator(estimator, predict_input_fn, hooks=None):
    """Given the input fn and estimator, yields one result."""
    for cnt, result in enumerate(estimator.predict(
            predict_input_fn, yield_single_examples=True, hooks=hooks)):
        if cnt % 1000 == 0:
            tf.logging.info("Processing example: %d" % cnt)
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        relevance_logits = float(result["relevance_logits"].flat[0])
        yield RawResult(unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits,
                        relevance_logits=relevance_logits)


def main(_):
  os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_lazy_compilation=false" #causes memory fragmentation for bert leading to OOM
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(FLAGS.rand_seed)
  np.random.seed(FLAGS.rand_seed)

  if FLAGS.horovod:
    hvd.init()
  if FLAGS.use_fp16:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"

  bert_config = modeling_func.BertConfig.from_json_file(FLAGS.bert_config_file)

  if bert_config.use_rel_pos_embeddings:
    tf.logging.info("Use relative position embeddings")
    tf.logging.info("max_rel_positions %d" % bert_config.max_rel_positions)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)
  model_dir = os.path.join(FLAGS.output_dir, "model_dir")
  tf.gfile.MakeDirs(model_dir)

  model_type = "roberta" if bert_config.roberta else "bert"
  if model_type == "bert":
      # Loads BERT tokenizer.
      tokenizer = tokenization.FullTokenizer(
          vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

      doc_token_processor = bert_doc_token_processor
      cls_tok = "[CLS]"
      sep_tok = "[SEP]"
      pad_token_id = 0
  elif model_type == "roberta":
      tokenizer = RobertaTokenizer(FLAGS.vocab_file, FLAGS.merges_file)
      doc_token_processor = roberta_doc_token_processor
      cls_tok = tokenizer.cls_token
      sep_tok = tokenizer.sep_token
      pad_token_id = tokenizer.pad_token_id

  else:
      raise ValueError("Unknown model_type %s" % model_type)

  # tokenizer = tokenization.FullTokenizer(
  #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  master_process = True
  training_hooks = []
  global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps
  hvd_rank = 0

  config = tf.ConfigProto()
  learning_rate = FLAGS.learning_rate
  if FLAGS.horovod:
      tf.logging.info("Multi-GPU training with TF Horovod")
      tf.logging.info("hvd.size() = %d hvd.rank() = %d", hvd.size(), hvd.rank())
      global_batch_size = FLAGS.train_batch_size * hvd.size() * FLAGS.num_accumulation_steps
      master_process = (hvd.rank() == 0)
      hvd_rank = hvd.rank()
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      if hvd.size() > 1:
          training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

  if FLAGS.do_predict and FLAGS.num_eval_split > 1:
      config.gpu_options.visible_device_list = str(FLAGS.eval_split_id)

  if FLAGS.use_xla:
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  run_config = tf.estimator.RunConfig(
      model_dir=model_dir if master_process else None,
      session_config=config,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps if master_process else None,
      keep_checkpoint_max=1)

  if master_process:
      tf.logging.info("***** Configuaration *****")
      for key in FLAGS.__flags.keys():
          tf.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
      tf.logging.info("**************************")

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank, FLAGS.save_checkpoints_steps))

  # Prepare Training Data
  if FLAGS.do_train:
    train_files = []
    for input_dir in FLAGS.tfrecord_dir.split(","):
        train_files.extend(tf.gfile.Glob(os.path.join(input_dir, "*.tf_record_*")))

    tf.logging.info("Reading tfrecords from %s" % "\n".join(train_files))
    num_train_steps = FLAGS.num_train_steps
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      hvd=None if not FLAGS.horovod else hvd,
      use_fp16=FLAGS.use_fp16,
      num_doc_features=FLAGS.max_num_doc_feature,
      max_short_answers=FLAGS.max_short_answers,
      seq_length=FLAGS.max_seq_length,
  )

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    tf.logging.info("  LR = %f", learning_rate)

    train_input_fn = doc_input_fn_builder(
        input_files=train_files,
        batch_size=FLAGS.train_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        hvd=None if not FLAGS.horovod else hvd,
        num_feature_per_doc=FLAGS.max_num_doc_feature,
        max_num_answers=FLAGS.max_short_answers,
    )

    train_start_time = time.time()
    estimator.train(input_fn=train_input_fn, hooks=training_hooks, max_steps=num_train_steps)
    train_time_elapsed = time.time() - train_start_time
    train_time_wo_overhead = training_hooks[-1].total_time
    avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
    ss_sentences_per_second = (num_train_steps - training_hooks[-1].skipped) * global_batch_size * 1.0 / train_time_wo_overhead

    if master_process:
        tf.logging.info("-----------------------------")
        tf.logging.info("Total Training Time = %0.2f for Sentences = %d", train_time_elapsed,
                        num_train_steps * global_batch_size)
        tf.logging.info("Total Training Time W/O Overhead = %0.2f for Sentences = %d", train_time_wo_overhead,
                        (num_train_steps - training_hooks[-1].skipped) * global_batch_size)
        tf.logging.info("Throughput Average (sentences/sec) with overhead = %0.2f", avg_sentences_per_second)
        tf.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
        tf.logging.info("-----------------------------")

  if FLAGS.do_predict and master_process:
    tf.logging.info("Using top-%d passages for infernece" % FLAGS.topk_for_infer)
    doc_eval_examples, _ = read_doc_squad_examples_from_generator(
      input_file=FLAGS.predict_file, is_training=False,
      keep_topk=FLAGS.topk_for_infer,
    )

    if FLAGS.use_doc_title:
        if doc_eval_examples[0].example_list[0].title is None:
            raise ValueError(
                "When use_doc_tile=True, doc_title is required in the input!")
        tf.logging.info("Inlcudes document tile for conversion.")

    # Perform evaluation on subset, useful for profiling
    # if FLAGS.num_eval_iterations is not None:
    #     eval_examples = eval_examples[:FLAGS.num_eval_iterations*FLAGS.predict_batch_size]

    start_index = 0
    end_index = len(doc_eval_examples)
    num_eval_examples = len(doc_eval_examples)
    hvd_rank = 0
    eval_input_dir = FLAGS.tfrecord_dir if FLAGS.tfrecord_dir else FLAGS.output_dir
    if FLAGS.num_eval_split > 1:
      # tmp_filenames = [os.path.join(
      #     FLAGS.output_dir, "eval.tf_record-{}".format(i)) for i in range(hvd.size())]
      # tmp_feat_filenames = [os.path.join(
      #     FLAGS.output_dir, "eval.converted_feature-{}".format(i)) for i in range(hvd.size())]
      tf_record_filename = os.path.join(
          eval_input_dir, "eval.tf_record-{}".format(FLAGS.eval_split_id))
      convert_feat_filename = os.path.join(
          eval_input_dir, "eval.converted_feature-{}".format(FLAGS.eval_split_id))

      # num_examples_per_rank = num_eval_examples // hvd.size()
      # remainder = num_eval_examples % hvd.size()
      num_examples_per_rank = num_eval_examples // FLAGS.num_eval_split
      remainder = num_eval_examples % FLAGS.num_eval_split
      hvd_rank = FLAGS.eval_split_id

      # if hvd.rank() < remainder:
      if FLAGS.eval_split_id < remainder:
        start_index = hvd_rank * (num_examples_per_rank+1)
        end_index = start_index + num_examples_per_rank + 1
      else:
        start_index = hvd_rank * num_examples_per_rank + remainder
        end_index = start_index + (num_examples_per_rank)
    else:
      tf_record_filename = os.path.join(FLAGS.output_dir, "eval.tf_record-0")
      convert_feat_filename = os.path.join(
          FLAGS.output_dir, "eval.converted_feature-0")

      # tmp_filenames = [os.path.join(FLAGS.output_dir, "eval.tf_record")]
      # tmp_feat_filenames = [os.path.join(FLAGS.output_dir, "eval.converted_feature")]

    def _flatten_doc_examples(doc_examples):
        """Flattens doc examples into passage examples."""
        return sum([de.example_list for de in doc_examples], [])

    eval_examples = _flatten_doc_examples(doc_eval_examples[start_index:end_index])
    del doc_eval_examples

    # tf_record_filename = tmp_filenames[hvd_rank]
    # convert_feat_filename = tmp_feat_filenames[hvd_rank]

    if (not tf.gfile.Exists(tf_record_filename)) or (
            not tf.gfile.Exists(convert_feat_filename)):
        eval_writer = FeatureWriter(
            filename=tf_record_filename,
            is_training=False)
        eval_features = []

        def append_feature(feature):
          eval_features.append(feature)
          eval_writer.process_feature(feature)

        unique_id_to_qid = collections.defaultdict()

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature,
            unique_id_to_qid=unique_id_to_qid,
            include_title=FLAGS.use_doc_title,
            cls_tok=cls_tok,
            sep_tok=sep_tok,
            pad_token_id=pad_token_id,
            doc_token_processor=doc_token_processor,
        )
        eval_writer.close()

        with open(convert_feat_filename, "w", encoding="utf8") as fout:
            for feature in eval_features:
                fout.write(feature.to_json())
                fout.write('\n')
    else:
        with open(convert_feat_filename, "r", encoding="utf8") as fin:
            eval_features = [
                InputFeatures.load_from_json(line)
                for line in fin if line.strip()
            ]

        unique_id_to_qid = dict([(feat.unique_id, feat.qid) for feat in eval_features])

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=tf_record_filename,
        batch_size=FLAGS.predict_batch_size,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    all_results = []
    eval_hooks = [LogEvalRunHook(FLAGS.predict_batch_size)]
    eval_start_time = time.time()
    # for result in estimator.predict(
    #     predict_input_fn, yield_single_examples=True, hooks=eval_hooks):
    #   if len(all_results) % 1000 == 0:
    #     tf.logging.info("Processing example: %d" % (len(all_results)))
    #   unique_id = int(result["unique_ids"])
    #   start_logits = [float(x) for x in result["start_logits"].flat]
    #   end_logits = [float(x) for x in result["end_logits"].flat]
    #   all_results.append(
    #       RawResult(
    #           unique_id=unique_id,
    #           start_logits=start_logits,
    #           end_logits=end_logits))
    all_results = [
        raw_result
        for raw_result in prediction_generator(estimator, predict_input_fn, hooks=eval_hooks)
    ]

    eval_time_elapsed = time.time() - eval_start_time
    eval_time_wo_overhead = eval_hooks[-1].total_time

    time_list = eval_hooks[-1].time_list
    time_list.sort()
    num_sentences = (eval_hooks[-1].count - eval_hooks[-1].skipped) * FLAGS.predict_batch_size

    avg = np.mean(time_list)
    cf_50 = max(time_list[:int(len(time_list) * 0.50)])
    cf_90 = max(time_list[:int(len(time_list) * 0.90)])
    cf_95 = max(time_list[:int(len(time_list) * 0.95)])
    cf_99 = max(time_list[:int(len(time_list) * 0.99)])
    cf_100 = max(time_list[:int(len(time_list) * 1)])
    ss_sentences_per_second = num_sentences * 1.0 / eval_time_wo_overhead

    tf.logging.info("-----------------------------")
    tf.logging.info("Total Inference Time = %0.2f for Sentences = %d", eval_time_elapsed,
                    eval_hooks[-1].count * FLAGS.predict_batch_size)
    tf.logging.info("Total Inference Time W/O Overhead = %0.2f for Sentences = %d", eval_time_wo_overhead,
                    (eval_hooks[-1].count - eval_hooks[-1].skipped) * FLAGS.predict_batch_size)
    tf.logging.info("Summary Inference Statistics")
    tf.logging.info("Batch size = %d", FLAGS.predict_batch_size)
    tf.logging.info("Sequence Length = %d", FLAGS.max_seq_length)
    tf.logging.info("Precision = %s", "fp16" if FLAGS.use_fp16 else "fp32")
    tf.logging.info("Latency Confidence Level 50 (ms) = %0.2f", cf_50 * 1000)
    tf.logging.info("Latency Confidence Level 90 (ms) = %0.2f", cf_90 * 1000)
    tf.logging.info("Latency Confidence Level 95 (ms) = %0.2f", cf_95 * 1000)
    tf.logging.info("Latency Confidence Level 99 (ms) = %0.2f", cf_99 * 1000)
    tf.logging.info("Latency Confidence Level 100 (ms) = %0.2f", cf_100 * 1000)
    tf.logging.info("Latency Average (ms) = %0.2f", avg * 1000)
    tf.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
    tf.logging.info("-----------------------------")

    unique_id_to_doc_score = compute_doc_norm_score(all_results, unique_id_to_qid)

    output_prediction_file = os.path.join(
        FLAGS.output_dir, "predictions-{0}.json".format(hvd_rank))
    output_nbest_file = os.path.join(
        FLAGS.output_dir, "nbest_predictions-{0}.json".format(hvd_rank))
    output_null_log_odds_file = os.path.join(
        FLAGS.output_dir, "null_odds-{0}.json".format(hvd_rank))

    write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      _compute_softmax, unique_id_to_doc_score,
                      model_type=model_type, tokenizer=tokenizer)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

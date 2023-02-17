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

import numpy as np
import six
import tensorflow as tf
from tensorflow.python.client import device_lib

import modeling_relpo as modeling_func
import optimization
import tokenization
import itertools
# from utils.create_squad_data import *
from utils.create_doc_qa_data import (
    read_doc_squad_examples_from_generator,
    read_squad_examples_from_generator,
    InputFeatureContainer,
    DocFeatureWriter,
    FeatureWriter,
    convert_doc_examples_to_doc_features,
    convert_examples_to_features,
    compute_doc_norm_score,
    BackgroundGenerator,
)
from utils.utils import LogEvalRunHook, LogTrainRunHook
from qa_utils.doc_qa_loss_helper import (
    answer_pos_loss,
    compute_masked_log_score,
    doc_pos_loss,
    doc_span_loss,
    par_pos_loss,
    par_span_loss,
)

from qa_utils.loss_helper import (
    compute_double_forward_loss_w_add_noise_correct_only,
    compute_double_forward_loss_w_add_noise,
    compute_double_forward_loss_w_add_noise_span,
    compute_vat_loss,
    compute_topk_vat_loss,
    compute_span_vat_loss,
    compute_approx_jacobian_loss,
    kl_divergence_w_log_prob,
    hellinger_distance_w_log_prob,
    js_divergence_w_log_prob,
)

from tensorflow.core.protobuf import rewriter_config_pb2

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "train_file_dir", None,
    "Directory with input files, comma separated or single directory.")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_bool("do_ema", False, "Whether to train with EMA.")
flags.DEFINE_float("ema_decay", 0.99, "The ema decay for training.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

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

flags.DEFINE_integer("rand_seed", 12345, "The random seed used")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("use_doc_title", False,
                  "Whether to include document title as part of query.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_integer("max_num_doc_feature", 12,
                     "Max number of document features allowed.")

flags.DEFINE_integer("topk_for_infer", 50,
                     "Max number of passages for inference.")

flags.DEFINE_integer("topk_for_train", 1000,
                     "Prefers top-ranked passages for training.")

flags.DEFINE_float("learning_rate", 5e-6, "The initial learning rate for Adam.")

flags.DEFINE_bool("use_trt", False, "Whether to use TF-TRT")

flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")
flags.DEFINE_float("num_train_epochs", 1.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("num_train_steps", 0,
                     "Total number of training steps to perform.")

flags.DEFINE_float("layerwise_lr_decay", -1.0,
                   "layerwise learning rate decay power.")

flags.DEFINE_float(
    "warmup_proportion", 0.05,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoint_steps", 1000,
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
    "n_best_size", 50,
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

flags.DEFINE_bool("single_pos_per_dupe", True,
                  "Whether to sample only one positive per duplicate.")

flags.DEFINE_bool("debug", False, "If true we process a tiny dataset.")

flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("manual_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("allreduce_post_accumulation", True, "Whether to all reduce after accumulation of N steps or after each step")
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

flags.DEFINE_integer("num_vat_est_iter", 1,
    "Integer for number of VAT estimations.")

flags.DEFINE_bool("accum_est", False,
    "Bool for whether to sum estimations.")

# Multi-passage losses.
flags.DEFINE_string("global_loss", "doc_pos-mml",
                    "The global multi-passage objective")

flags.DEFINE_string("vat_type", "global_local",
                    "The global multi-passage objective")

# Single-passage losses.
flags.DEFINE_string("local_loss", "None",
                    "The local passage-level objective")

flags.DEFINE_float(
    "teacher_temperature", 1.0,
    "A knob to sharpen or flatten the teacher distribution.")

flags.DEFINE_float(
    "local_obj_alpha", 0.0,
    "Trades off between clean global and noisy local objectives.")


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
      token_type_ids=None if bert_config.roberta else segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      compute_type=tf.float16 if FLAGS.manual_fp16 else tf.float32,
      scope=scope,
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

  start_logits = tf.cast(start_logits, tf.float32)
  end_logits = tf.cast(end_logits, tf.float32)

  return (start_logits, end_logits, relevance_logits, model)


def get_var_with_name(var_name):
    name = var_name + ":0"
    # name = var_name
    return tf.get_default_graph().get_tensor_by_name(name)


class DocQAModel(object):
    """Document QA model."""

    def __init__(self, bert_config, mode):
        # Makes those variables as local variables.
        self.max_seq_length = FLAGS.max_seq_length
        self.max_num_answers = FLAGS.max_short_answers
        self.max_num_answer_strings = FLAGS.max_num_answer_strings

        self._inputs, self._outputs, self._model = self.build_model(
            mode, bert_config)

        self._fetch_var_names = []
        self._fetch_var_names.append('loss_to_opt')
        if mode != 'TRAIN':
            self._fetch_var_names.append('start_logits', 'end_logits')

    def check_fetch_var(self):
        """Checks whether all variables to fetch are in the output dict."""
        # Checks whether required variables are in the outputs_.
        for var_name in self._fetch_var_names:
            if var_name not in self._outputs:
                raise ValueError(
                    '{0} is not in the output list'.format(var_name))

    def build_model(self, mode, bert_config, use_one_hot_embeddings=False):
        """Builds the model based on BERT."""
        input_ids = tf.placeholder(
            tf.int32, name='input_ids', shape=[None, self.max_seq_length]
        )
        input_mask = tf.placeholder(
            tf.int32, name='input_mask', shape=[None, self.max_seq_length]
        )
        segment_ids = tf.placeholder(
            tf.int32, name="segment_ids", shape=[None, self.max_seq_length]
        )

        is_training = (mode == 'TRAIN')

        (start_logits, end_logits, relevance_logits, model) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )


        inputs = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
        }

        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }

        return inputs, outputs, model

    def build_loss(self):
        """Builds loss variables."""

        start_logits = self._outputs["start_logits"]
        end_logits = self._outputs["end_logits"]

        if FLAGS.teacher_temperature < 1.0:
            # Performs dampening on the logits.
            tf.logging.info("Dampening the logits with temperature %s" %
                            FLAGS.teacher_temperature)
            start_logits *= FLAGS.teacher_temperature
            end_logits *= FLAGS.teacher_temperature

        input_shape = modeling_func.get_shape_list(
            self._inputs['input_ids'], expected_rank=2)
        batch_size, seq_length = input_shape[0], input_shape[1]

        # Builds input placeholder variables.
        start_positions_list = tf.placeholder(
            tf.int32, name="start_position_list",
            shape=[None, self.max_num_answers]
        )
        end_positions_list = tf.placeholder(
            tf.int32, name="end_position_list",
            shape=[None, self.max_num_answers]
        )
        answer_positions_mask = tf.placeholder(
            tf.int32, name="answer_positions_mask",
            shape=[None, self.max_num_answers]
        )

        answer_index_list = tf.placeholder(
            tf.int32, name="answer_index_list",
            shape=[None, self.max_num_answers]
        )
        notnull_answer_mask = tf.placeholder(
            tf.int32, name="notnull_answer_mask",
            shape=[None, self.max_num_answers]
        )

        # Builds the input-variable map.
        self._inputs['start_positions_list'] = start_positions_list
        self._inputs['end_positions_list'] = end_positions_list
        self._inputs['answer_positions_mask'] = answer_positions_mask
        self._inputs['answer_index_list'] = answer_index_list
        self._inputs["notnull_answer_mask"] = notnull_answer_mask

        total_loss = 0.0
        tf.logging.info('Using margmalized loss for training')

        global_loss = FLAGS.global_loss
        local_loss = FLAGS.local_loss

        positive_par_mask = tf.reduce_max(notnull_answer_mask, axis=-1)

        if global_loss == 'doc_pos-mml':
            tf.logging.info("Using doc_pos-mml for global objective")
            doc_answer_positions_mask = answer_positions_mask * notnull_answer_mask
            total_loss += doc_pos_loss(
                start_logits, start_positions_list, end_logits,
                end_positions_list, doc_answer_positions_mask,
                positive_par_mask, seq_length, loss_type='doc_mml'
            )

        if global_loss == 'doc_span-mml':
            tf.logging.info("Using doc_span-mml for global objective")
            doc_answer_positions_mask = answer_positions_mask * notnull_answer_mask
            total_loss += doc_span_loss(
                start_logits, start_positions_list, end_logits,
                end_positions_list, doc_answer_positions_mask,
                positive_par_mask, loss_type='doc_mml'
            )

        if global_loss == 'doc_pos-hard_em':
            tf.logging.info("Using doc_pos-hard_em for global objective")
            doc_answer_positions_mask = answer_positions_mask * notnull_answer_mask
            # total_loss += doc_pos_loss(
            #     start_logits, start_positions_list, end_logits,
            #     end_positions_list, doc_answer_positions_mask,
            #     positive_par_mask, seq_length, loss_type='doc_hard_em'
            # )
            total_loss += doc_span_loss(
                start_logits, start_positions_list, end_logits,
                end_positions_list, doc_answer_positions_mask,
                positive_par_mask, loss_type='doc_hard_em'
            )


        if global_loss == 'pos_par_doc_pos-mml':
            tf.logging.info("Using pos_par_doc_pos-mml for global objective")
            total_loss += doc_pos_loss(
                start_logits, start_positions_list, end_logits,
                end_positions_list, answer_positions_mask,
                positive_par_mask, seq_length, loss_type='positive_par_doc_mml'
            )

        if global_loss == 'pos_par_doc_span-mml':
            tf.logging.info("Using pos_par_doc_span-mml for global objective")
            total_loss += doc_span_loss(
                start_logits, start_positions_list, end_logits,
                end_positions_list, answer_positions_mask,
                positive_par_mask, loss_type='positive_par_doc_mml'
            )

        if global_loss == 'pos_par_doc_pos-hard_em':
            tf.logging.info("Using pos_par_doc_pos-hard_em for global objective")
            # total_loss += doc_pos_loss(
            #     start_logits, start_positions_list, end_logits,
            #     end_positions_list, answer_positions_mask,
            #     positive_par_mask, seq_length,
            #     loss_type='positive_par_doc_hard_em'
            # )
            total_loss += doc_span_loss(
                start_logits, start_positions_list, end_logits,
                end_positions_list, answer_positions_mask,
                positive_par_mask, loss_type='positive_par_doc_hard_em'
            )

        method = global_loss.split("-")[1]
        if method == "pd":
            loss_type = global_loss.split("-")[-1]
            doc_answer_positions_mask = answer_positions_mask * notnull_answer_mask
            tf.logging.info("Using doc_pos-pd for global objective")
            total_loss += doc_pos_loss(
                start_logits, start_positions_list, end_logits,
                end_positions_list, doc_answer_positions_mask,
                positive_par_mask, seq_length,
                loss_type='pd', logit_temp=FLAGS.teacher_temperature,
                pd_loss=loss_type,
            )

        # Paragraph-level positive paragraph correct span-based marginalization loss.
        if FLAGS.local_obj_alpha > 0.0:
            if local_loss == 'pos_loss':
                tf.logging.info("Using pos_loss for local objective")
                total_loss += FLAGS.local_obj_alpha * par_pos_loss(
                    start_logits, start_positions_list, end_logits,
                    end_positions_list, answer_positions_mask, seq_length,
                    loss_type='par_mml'
                )
            elif local_loss == 'span_hard_em':
                tf.logging.info("Using span_hard_em for local objective")
                total_loss += FLAGS.local_obj_alpha * par_span_loss(
                    start_logits, start_positions_list, end_logits,
                    end_positions_list, answer_positions_mask,
                    loss_type='par_hard_em'
                )
            elif local_loss == 'span_loss':
                tf.logging.info("Using span_loss for local objective")
                total_loss += FLAGS.local_obj_alpha * par_span_loss(
                    start_logits, start_positions_list, end_logits,
                    end_positions_list, answer_positions_mask,
                    loss_type='par_mml'
                )

            elif local_loss == "pos_hard_em":
                tf.logging.info("Using pos_hard_em for local objective")
                total_loss += FLAGS.local_obj_alpha * par_pos_loss(
                    start_logits, start_positions_list, end_logits,
                    end_positions_list, answer_positions_mask, seq_length,
                    loss_type='par_hard_em'
                )
            else:
                tf.logging.info("Unknown local_loss %s" % local_loss)

        clean_loss = total_loss
        if FLAGS.vat_reg_rate > 0.0:
            loss_items = FLAGS.double_forward_loss.split("-")
            if len(loss_items) == 2:
                vat_type = loss_items[0]
                loss_type = loss_items[1]
            elif len(loss_items) == 1:
                vat_type = "vanilla"
                loss_type = loss_items[0]
            else:
                raise ValueError(
                    "Unknown loss config for vat %s" % FLAGS.double_forward_loss)

            if vat_type == "vanilla":
                tf.logging.info("Using Vanilla-VAT")
                vat_loss = compute_vat_loss(
                    start_logits, end_logits, self._model,
                    noise_epsilon=FLAGS.noise_epsilon,
                    noise_normalizer=FLAGS.noise_normalizer,
                    loss_type=loss_type,
                    vat_type=FLAGS.vat_type,
                    accum_est=FLAGS.accum_est,
                    num_est_iter=FLAGS.num_vat_est_iter,
                )
            elif vat_type == "span":
                tf.logging.info("Using Span-VAT")
                vat_loss = compute_span_vat_loss(
                    start_logits, end_logits, self._model,
                    noise_epsilon=FLAGS.noise_epsilon,
                    noise_normalizer=FLAGS.noise_normalizer,
                    loss_type=loss_type,
                    k=FLAGS.n_best_size,
                    span_ub=FLAGS.max_answer_length,
                    vat_type=FLAGS.vat_type,
                    valid_span_only=True,
                )
            elif vat_type == "topk":
                tf.logging.info("Using Topk-VAT")
                # Both one-hot tensors of shape [batch_size, 1, seq_length].
                # start_one_hot_mask = tf.reduce_max(
                #     tf.one_hot(start_positions_list, depth=seq_length, dtype=tf.float32),
                #     axis=1, keepdims=True)
                # end_one_hot_mask = tf.reduce_max(
                #     tf.one_hot(end_positions_list, depth=seq_length, dtype=tf.float32),
                #     axis=1, keepdims=True)

                vat_loss = compute_topk_vat_loss(
                    start_logits, end_logits, self._model,
                    noise_epsilon=FLAGS.noise_epsilon,
                    noise_normalizer=FLAGS.noise_normalizer,
                    loss_type=loss_type,
                    vat_type=FLAGS.vat_type,
                    top_k=FLAGS.n_best_size,
                    accum_est=FLAGS.accum_est,
                    num_est_iter=FLAGS.num_vat_est_iter,
                )
            else:
                raise ValueError("Unknown VAT type %s" % vat_type)

            total_loss += FLAGS.vat_reg_rate * vat_loss
        else:
            vat_loss = tf.constant(0.0)

        if FLAGS.double_forward_reg_rate > 0.0:
          k = 20
          double_forward_loss = compute_double_forward_loss_w_add_noise_span(
              start_logits, end_logits, self._model,
              loss_type=FLAGS.double_forward_loss,
              noise_epsilon=FLAGS.noise_epsilon,
              noise_normalizer=FLAGS.noise_normalizer,
              alpha=FLAGS.kl_alpha,
              beta=FLAGS.kl_beta,
              k=k,
              flat_logits=True,
          )
          total_loss += FLAGS.double_forward_reg_rate * double_forward_loss
        else:
          double_forward_loss = tf.constant(0.0)

        self._outputs['loss_to_opt'] = total_loss
        self.logging_vars = [
            "total_loss",
            "clean_loss",
            "vat_loss",
            "scaled_vat_loss",
            "double_forward_loss",
            "scaled_double_forward_loss",
        ]
        self.logging_hook = {
            "total_loss": total_loss,
            "clean_loss": clean_loss,
            "vat_loss": vat_loss,
            "scaled_vat_loss": FLAGS.vat_reg_rate * vat_loss,
            "double_forward_loss": double_forward_loss,
            "scaled_double_forward_loss": FLAGS.double_forward_reg_rate * double_forward_loss,
        }

    def build_opt_op(self, learning_rate, num_train_steps, num_warmup_steps,
                     use_fp16=False, hvd=None, manual_fp16=False,
                     num_accumulation_steps=1, optimizer="adam",
                     allreduce_post_accumulation=True,
                     n_transformer_layers=None,
                     layerwise_lr_decay_power=-1.0):
        """Builds optimization operator for the model."""
        loss_to_opt = self._outputs['loss_to_opt']

        train_op = optimization.create_optimizer(
            loss_to_opt,
            learning_rate,
            num_train_steps,
            num_warmup_steps,
            hvd=hvd,
            manual_fp16=manual_fp16,
            use_fp16=use_fp16,
            num_accumulation_steps=num_accumulation_steps,
            layerwise_lr_decay_power=layerwise_lr_decay_power,
            n_transformer_layers=n_transformer_layers,
            optimizer_type=optimizer,
            allreduce_post_accumulation=allreduce_post_accumulation
        )
        if FLAGS.do_ema:
            ema = tf.train.ExponentialMovingAverage(
                decay=FLAGS.ema_decay,
            )

            tvars = tf.trainable_variables()
            with tf.control_dependencies([train_op]):
                ema_train_op = ema.apply(tvars)

            return ema_train_op

        return train_op
        # return optimization_orig.create_optimizer(
        #     loss_to_opt, learning_rate, num_train_steps, num_warmup_steps,
        #     use_tpu
        # )

    def _run_model(self, session, feed_dict, opt_op, log_hook=False):
        """Performans a forward and backward pass of the model."""

        # fetches = [get_var_with_name(var_name)
        fetches = [self._outputs[var_name]
                   for var_name in self._fetch_var_names]

        if log_hook:
            fetches += [
                self.logging_hook[var_name] for var_name in self.logging_vars]
                # get_var_with_name(var_name) for var_name in self.logging_vars]

        fetches.append(opt_op)

        all_outputs = session.run(fetches, feed_dict)

        if log_hook:
            offset = len(self._fetch_var_names)
            fetched_var_dict = dict([
                (var_name, all_outputs[idx])
                for idx, var_name in enumerate(self._fetch_var_names)
            ] + [
                (var_name, all_outputs[offset + idx])
                for idx, var_name in enumerate(self.logging_vars)
            ])
        else:
            fetched_var_dict = dict([
                (var_name, all_outputs[idx])
                for idx, var_name in enumerate(self._fetch_var_names)
            ])

        return fetched_var_dict

    def _build_feed_dict(self, inputs_dict):
        """Builds feed dict for inputs."""
        feed_dict_list = []
        for input_name, input_var in self._inputs.items():
            if input_name not in inputs_dict:
                raise ValueError('Missing input_name: {0}'.format(input_name))
            feed_dict_list.append((input_var, inputs_dict[input_name]))
            # feed_dict_list.append((input_var, get_var_with_name(input_name)))
        return dict(feed_dict_list)

    def one_step(self, session, inputs_dict, opt_op, log_hook=False):
        """Trains, evaluates, or infers the model with one batch of data."""
        feed_dict = self._build_feed_dict(inputs_dict)
        fetched_dict = self._run_model(session, feed_dict, opt_op,
                                       log_hook=log_hook)

        return fetched_dict

    def initialize_from_checkpoint(self, init_checkpoint):
        """Initializes model variables from init_checkpoint."""
        variables_to_restore = tf.trainable_variables()

        if init_checkpoint:
            tf.logging.info(
                "Initializing the model from {0}".format(init_checkpoint))
            (assignment_map, initialized_variable_names
             ) = modeling_func.get_assignment_map_from_checkpoint(
                 variables_to_restore, init_checkpoint
             )

            # TODO(chenghao): Currently, no TPU is supported.
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in variables_to_restore:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)


def run_epoch(model, session, data_container, opt_op, mode,
              model_saver=None, verbose=True, eval_func=None,
              queue_size=100,
              num_train_steps=None, is_master=False, log_n_iter=100):
    """Runs one epoch over the data."""
    start_time = time.time()
    iter_count = 1

    process_sample_cnt = 0

    # Starts a new epoch.
    data_container.start_epoch()

    total_loss = 0.0
    model_dir = os.path.join(FLAGS.output_dir, "model_dir")

    def _log_it(it):
        return (it % log_n_iter) == 0 and is_master and verbose

    def _print_log_hooks(log_vars, val_dict):
        return ", ".join([
            "{0}: {1:.5f}".format(var, val_dict[var])
            for var in log_vars
        ])

    last_global_step = int(session.run(tf.train.get_global_step()))
    for iter_count, feature_dict in enumerate(
            BackgroundGenerator(data_container, max_prefetch=queue_size)):
        log_it = _log_it(iter_count)
        fetched_dict = model.one_step(session, feature_dict, opt_op,
                                      log_hook=log_it)
        total_loss += fetched_dict['loss_to_opt']

        process_sample_cnt += feature_dict['num_sample']

        if log_it:
            tf.logging.info(
                'iter {:d}:, {:.3f} examples per second'.format(
                    iter_count,
                    process_sample_cnt / (time.time() - start_time)
                )
            )
            tf.logging.info(
                _print_log_hooks(model.logging_vars, fetched_dict)
            )

        global_step = int(session.run(tf.train.get_global_step()))
        if (is_master and model_saver is not None and
                global_step > last_global_step and
                global_step % FLAGS.save_checkpoint_steps == 0):
            tf.logging.info("save checkpoints %d into %s" % (
                global_step, model_dir
            ))
            model_saver.save(
                session,
                os.path.join(model_dir, 'model.ckpt'),
                global_step=global_step
            )
            last_global_step = global_step

        if num_train_steps is not None and global_step >= num_train_steps:
            tf.logging.info(
                'Reaches the num_train_steps {0}'.format(num_train_steps))
            break

    if is_master:
        tf.logging.info(
            'time for one epoch: {:.3f} secs'.format(time.time() - start_time)
        )
        tf.logging.info('iters over {0} num of samples'.format(process_sample_cnt))

    eval_metric = {'total_loss': total_loss}
    output_dict = {}

    return total_loss, eval_metric, output_dict


def train_model(session_config, train_data_container, bert_config, learning_rate,
                num_train_steps, num_warmup_steps, batch_size, shuffle_data,
                init_checkpoint, rand_seed=12345, chunk_size=3000, hvd=None,
                hooks=None, is_master=False, log_per_iter=1000):
    """ Training wrapper function."""
    model_dir = os.path.join(FLAGS.output_dir, "model_dir")
    tf.set_random_seed(rand_seed)
    np.random.seed(rand_seed)

    model_saver = None
    # with tf.Graph().as_default(), tf.Session(config=session_config) as session:
    #     model = DocQAModel(bert_config, 'TRAIN')

    #     # This is needed for both TRAIN and EVAL.
    #     model.build_loss()
    #     model.check_fetch_var()

    #     # This operation is only needed for TRAIN phase.
    #     opt_op = model.build_opt_op(
    #         learning_rate, num_train_steps, num_warmup_steps,
    #         hvd=hvd,
    #         manual_fp16=FLAGS.manual_fp16,
    #         use_fp16=FLAGS.use_fp16,
    #         num_accumulation_steps=FLAGS.num_accumulation_steps,
    #         optimizer=FLAGS.optimizer_type,
    #         allreduce_post_accumulation=FLAGS.allreduce_post_accumulation,
    #     )

    #     # Loads pretrain model parameters if specified.
    #     if is_master and init_checkpoint:
    #         model_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
    #         model.initialize_from_checkpoint(init_checkpoint)

    #     tf.compat.v1.train.get_or_create_global_step()
    #     session.run(tf.global_variables_initializer())

    #     if hvd:
    #         session.run(hvd.broadcast_global_variables(0))

    #     tf.get_default_graph().finalize()

    #     # for it in range(num_train_epochs):
    #     for it in range(1000):
    #         if is_master:
    #             tf.logging.info('Train Iter {0}'.format(it))

    #         _, train_metric, _ = run_epoch(
    #             model, session, train_data_container,
    #             opt_op, 'TRAIN', eval_func=None,
    #             model_saver=model_saver,
    #             num_train_steps=num_train_steps,
    #             is_master=is_master,
    #             log_n_iter=log_per_iter,
    #         )

    #         global_step = int(session.run(tf.train.get_global_step()))
    #         if is_master:
    #             tf.logging.info('\n'.join([
    #                 'train {}: {:.3f}'.format(metric_name, metric_val)
    #                 for metric_name, metric_val in train_metric.items()
    #             ]))
    #             model_saver.save(
    #                 session,
    #                 os.path.join(model_dir, 'model.ckpt'),
    #                 global_step=global_step
    #             )

    #         if num_train_steps is not None and global_step >= num_train_steps:
    #             break

    #     if hvd:
    #         tf.logging.info("Training done for %d" % hvd.rank())
    #     else:
    #         tf.logging.info('Training model done!')

    #     return True

    if bert_config.short_cut:
        tf.logging.info("Using short-cut Transformers")

    tf.train.get_or_create_global_step()
    model = DocQAModel(bert_config, 'TRAIN')
    model.build_loss()
    model.check_fetch_var()

    # This operation is only needed for TRAIN phase.
    opt_op = model.build_opt_op(
        learning_rate, num_train_steps, num_warmup_steps,
        hvd=hvd,
        manual_fp16=FLAGS.manual_fp16,
        use_fp16=FLAGS.use_fp16,
        num_accumulation_steps=FLAGS.num_accumulation_steps,
        optimizer=FLAGS.optimizer_type,
        layerwise_lr_decay_power=FLAGS.layerwise_lr_decay,
        n_transformer_layers=bert_config.num_hidden_layers,
        allreduce_post_accumulation=FLAGS.allreduce_post_accumulation,
    )

    # Loads pretrain model parameters if specified.
    if is_master and init_checkpoint:
        model.initialize_from_checkpoint(init_checkpoint)

    # stop_hook = tf.train.StopAtStepHook(last_step=num_train_steps)
    # hooks.append(stop_hook)
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=model_dir if is_master else None,
            save_checkpoint_steps=FLAGS.save_checkpoint_steps if is_master else None,
            config=session_config, hooks=hooks,
            log_step_count_steps=log_per_iter) as session:
        tf.get_default_graph().finalize()

        while not session.should_stop():
            for it in range(1000):
                if is_master:
                    tf.logging.info('Train Iter {0}'.format(it))

                _, train_metric, _ = run_epoch(
                    model, session, train_data_container,
                    opt_op, 'TRAIN', eval_func=None,
                    model_saver=None,
                    num_train_steps=num_train_steps,
                    is_master=is_master,
                    log_n_iter=log_per_iter,
                )

                if is_master:
                    tf.logging.info('\n'.join([
                        'train {}: {:.3f}'.format(metric_name, metric_val)
                        for metric_name, metric_val in train_metric.items()
                    ]))

                global_step = session.run(tf.train.get_global_step())
                if num_train_steps is not None and global_step >= num_train_steps:
                    break


            if hvd:
                tf.logging.info("Training done for %d" % hvd.rank())
            else:
                tf.logging.info('Training model done!')

            break

    return True


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
        use_one_hot_embeddings=use_one_hot_embeddings)

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

      if FLAGS.global_loss != "None":
        tf.logging.info("Using document-level loss!")
        # doc_feature_size = seq_length * num_doc_features
        # position_offset = tf.tile(tf.reshape(
        #     tf.range(0, doc_feature_size, seq_length, dtype=tf.int32),
        #     [1, num_doc_features]), [batch_size, 1])
        # position_offset = reshape_doc_feature(
        #     position_offset, batch_size, 1, num_doc_features)

        flat_start_logits = flat_doc_feature(
            start_logits, batch_size, seq_length, num_doc_features)
        flat_end_logits = flat_doc_feature(
            end_logits, batch_size, seq_length, num_doc_features)

        # flat_start_positions = flat_doc_feature(
        #   start_positions + position_offset, batch_size, max_short_answers, num_doc_features)
        # flat_end_positions = flat_doc_feature(
        #   end_positions + position_offset, batch_size, max_short_answers, num_doc_features)
        # flat_seq_length = seq_length * num_doc_features


        z_start = tf.reduce_logsumexp(flat_start_logits, axis=-1, keepdims=True)
        z_end = tf.reduce_logsumexp(flat_end_logits, axis=-1, keepdims=True)

        if FLAGS.global_loss == "doc_pos-mml":
            flat_answer_positions_mask = features["answer_positions_mask"] * features["notnull_answer_mask"]
        elif FLAGS.global_loss == "pos_par_pos-mml":
            flat_answer_positions_mask = features["answer_positions_mask"]

        answer_positions_mask = reshape_doc_feature(
            flat_answer_positions_mask,
            batch_size, max_short_answers, num_doc_features)

        masked_start_logits = compute_masked_log_score(
            start_logits,
            start_positions,
            answer_positions_mask,
            seq_length
        )
        masked_end_logits = compute_masked_log_score(
            end_logits,
            end_positions,
            answer_positions_mask,
            seq_length
        )

        if FLAGS.global_loss == "doc_pos-mml":
            tf.logging.info("Using document-level MML POS loss!")
            # The 1 dim is batch_size.
            flat_masked_start_logits = flat_doc_feature(
                masked_start_logits, batch_size, seq_length, num_doc_features)
            flat_masked_end_logits = flat_doc_feature(
                masked_end_logits, batch_size, seq_length, num_doc_features)

            mml_start_loss = z_start - tf.reduce_logsumexp(
                flat_masked_start_logits, axis=-1, keepdims=True)
            mml_end_loss = z_end - tf.reduce_logsumexp(
                flat_masked_end_logits, axis=-1, keepdims=True)
        elif FLAGS.global_loss == "pos_par_pos-mml":
            tf.logging.info("Using document-level positive par MML POS loss!")
            notnull_answer_mask = reshape_doc_feature(
                features["notnull_answer_mask"],
                batch_size, max_short_answers, num_doc_features)
            pos_par_flag = flat_doc_feature(
                tf.cast(tf.reduce_max(
                    notnull_answer_mask, axis=-1, keepdims=True), tf.float32),
                batch_size, 1, num_doc_features)

            mml_start_loss = pos_par_flag * (
                z_start - flat_doc_feature(
                    tf.reduce_logsumexp(masked_start_logits, axis=-1, keepdims=True),
                    batch_size, 1, num_doc_features))

            mml_end_loss = pos_par_flag * (
                z_end - flat_doc_feature(
                    tf.reduce_logsumexp(masked_end_logits, axis=-1, keepdims=True),
                    batch_size, 1, num_doc_features))
        else:
            raise ValueError("Unknown global_loss %s" % FLAGS.global_loss)

        global_loss = tf.reduce_sum(mml_start_loss + mml_end_loss) / 2.0
        total_loss += global_loss

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
          manual_fp16=FLAGS.manual_fp16, use_fp16=FLAGS.use_fp16,
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
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
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
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      prob_transform_func, unique_id_to_doc_score):
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
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "doc_score"])

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
                            doc_score=unique_id_to_doc_score[result.unique_id]))

        if FLAGS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                    doc_score=null_doc_score))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "doc_score"])

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

                if FLAGS.model_type == "bert":
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")
                else:
                    # Converts back to normal string.
                    tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)

                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

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
                    doc_score=pred.doc_score))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if FLAGS.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit,
                        doc_score=null_doc_score))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, doc_score=0.0))

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
                text="empty", start_logit=0.0, end_logit=0.0, doc_score=0.0
                )

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
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
    if not FLAGS.train_file_dir:
      raise ValueError(
          "If `do_train` is True, then `train_file_dir` must be specified.")
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
  # if bert_config.roberta and FLAGS.merges_file is None:
  #     raise ValueError("When using roberta, merges file must be specified")
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
        yield RawResult(unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits)


def setup_xla_flags():
  # causes memory fragmentation for bert leading to OOM
  if os.environ.get("TF_XLA_FLAGS", None) is not None:
    try:
      os.environ["TF_XLA_FLAGS"] += " --tf_xla_enable_lazy_compilation=false"
    except: #mpi 4.0.2 causes syntax error for =
      os.environ["TF_XLA_FLAGS"] += " --tf_xla_enable_lazy_compilation false"
  else:
    try:
      os.environ["TF_XLA_FLAGS"] = " --tf_xla_enable_lazy_compilation=false"
    except:
      os.environ["TF_XLA_FLAGS"] = " --tf_xla_enable_lazy_compilation false"


def main(_):
  # os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_lazy_compilation=false" #causes memory fragmentation for bert leading to OOM
  # if FLAGS.use_fp16:
  #   os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  setup_xla_flags()

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(FLAGS.rand_seed)
  np.random.seed(FLAGS.rand_seed)

  if FLAGS.horovod:
    import horovod.tensorflow as hvd
    hvd.init()

  bert_config = modeling_func.BertConfig.from_json_file(FLAGS.bert_config_file)

  if bert_config.use_rel_pos_embeddings:
    tf.logging.info("Use relative position embeddings")
    tf.logging.info("max_rel_positions %d" % bert_config.max_rel_positions)

  if bert_config.roberta:
    tf.logging.info("Using RoBERTa for training")

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)
  model_dir = os.path.join(FLAGS.output_dir, "model_dir")
  tf.gfile.MakeDirs(model_dir)

  model_type = "roberta" if bert_config.roberta else "bert"
  # if model_type == "bert":
  #     # Loads BERT tokenizer.
  #     tokenizer = tokenization.FullTokenizer(
  #         vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  #     doc_token_processor = bert_doc_token_processor
  #     cls_tok = "[CLS]"
  #     sep_tok = "[SEP]"
  #     pad_token_id = 0
  # elif model_type == "roberta":
  #     tokenizer = RobertaTokenizer(FLAGS.vocab_file, FLAGS.merges_file)
  #     doc_token_processor = roberta_doc_token_processor
  #     cls_tok = tokenizer.cls_token
  #     sep_tok = tokenizer.sep_token
  #     pad_token_id = tokenizer.pad_token_id

  # else:
  #     raise ValueError("Unknown model_type %s" % model_type)

  # tokenizer = tokenization.FullTokenizer(
  #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  master_process = True
  training_hooks = []
  global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps
  hvd_rank = 0
  hvd_size = 1

  config = tf.compat.v1.ConfigProto()
  learning_rate = FLAGS.learning_rate
  if FLAGS.horovod:
      hvd_size = int(hvd.size())
      config.gpu_options.visible_device_list = str(hvd.local_rank())
      tf.logging.info("Multi-GPU training with TF Horovod")
      tf.logging.info("hvd.size() = %d hvd.rank() = %d", hvd.size(), hvd.rank())
      global_batch_size = FLAGS.train_batch_size * hvd.size() * FLAGS.num_accumulation_steps
      master_process = (hvd.rank() == 0)
      hvd_rank = hvd.rank()
      if hvd.size() > 1:
          training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

  config.gpu_options.allow_growth = True

  # if FLAGS.use_fp16:
  #   config.graph_options.rewrite_options.auto_mixed_precision = 1
  if FLAGS.use_xla:
    config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
    config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT
    if FLAGS.use_fp16:
      tf.enable_resource_variables()

  if master_process:
      tf.logging.info("***** Configuaration *****")
      for key in FLAGS.__flags.keys():
          tf.logging.info('  {}: {}'.format(key, getattr(FLAGS, key)))
      tf.logging.info("**************************")

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  training_hooks.append(LogTrainRunHook(global_batch_size, hvd_rank, FLAGS.save_checkpoint_steps))

  # Prepare Training Data
  num_train_files = 0
  if FLAGS.do_train:
    train_files = []
    for input_dir in FLAGS.train_file_dir.split(","):
        train_files.extend(tf.gfile.Glob(os.path.join(input_dir, "*.tf_record_*")))

    tf.logging.info("Reading tfrecords from %s" % "\n".join(train_files))

    num_train_files = len(train_files)

  if FLAGS.do_train:
    file_index = hvd_rank
    if FLAGS.horovod:
        if hvd_size > num_train_files:
            tf.logging.info("Num of train files is less than num of gpus")
            file_index = hvd_rank % num_train_files
        elif (num_train_files % hvd_size) > 0:
            raise ValueError("Uneven number of train files")

    group_func = lambda x: x[0] % hvd_size
    indexed_train_files = [(ii, ff) for ii, ff in enumerate(train_files)]
    grouped_train_files = [
        [x[1] for x in gp]
        for _, gp in itertools.groupby(
                sorted(indexed_train_files, key=group_func), key=group_func)]

    train_data_container = InputFeatureContainer(
        grouped_train_files[file_index], FLAGS.max_num_doc_feature, FLAGS.train_batch_size, True,
        FLAGS.rand_seed, single_pos_per_dupe=FLAGS.single_pos_per_dupe,
        allow_null_doc=(not FLAGS.filter_null_doc),
        topk_for_train=FLAGS.topk_for_train,
    )

    num_train_steps = int(FLAGS.num_train_steps)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    tf.logging.info("  LR = %f", learning_rate)

    train_start_time = time.time()
    train_model(config, train_data_container, bert_config, learning_rate,
                num_train_steps, num_warmup_steps, FLAGS.train_batch_size, True,
                FLAGS.init_checkpoint, rand_seed=FLAGS.rand_seed,
                hvd=hvd if FLAGS.horovod else None, hooks=training_hooks,
                is_master=master_process)

    train_time_elapsed = time.time() - train_start_time
    # train_time_wo_overhead = training_hooks[-1].total_time
    avg_sentences_per_second = num_train_steps * global_batch_size * 1.0 / train_time_elapsed
    # ss_sentences_per_second = (num_train_steps - training_hooks[-1].skipped) * global_batch_size * 1.0 / train_time_wo_overhead

    if master_process:
        tf.logging.info("-----------------------------")
        tf.logging.info("Total Training Time = %0.2f for Sentences = %d", train_time_elapsed,
                        num_train_steps * global_batch_size)
        # tf.logging.info("Total Training Time W/O Overhead = %0.2f for Sentences = %d", train_time_wo_overhead,
        #                 (num_train_steps - training_hooks[-1].skipped) * global_batch_size)
        tf.logging.info("Throughput Average (sentences/sec) with overhead = %0.2f", avg_sentences_per_second)
        # tf.logging.info("Throughput Average (sentences/sec) = %0.2f", ss_sentences_per_second)
        tf.logging.info("-----------------------------")

    if FLAGS.do_predict:
        tf.logging.info("Prediction not supported!")


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

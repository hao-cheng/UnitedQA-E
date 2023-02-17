#!/usr/bin/env python3
"""Loss help functions."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import modeling
import sys


def batch_gather(params, indices):
    """Performs batch_gather."""
    bs, nv = modeling.get_shape_list(indices, expected_rank=2)
    _, batch_offset = modeling.get_shape_list(params, expected_rank=2)

    # A [batch_size, num_answers]-sized tensor.
    offset = tf.tile(tf.reshape(
        tf.range(bs * batch_offset, delta=batch_offset, dtype=tf.int32),
        shape=[bs, 1]), [1, nv])

    f_indices = tf.reshape(indices + offset, shape=[-1])
    f_vals = tf.gather(tf.reshape(params, shape=[-1]), f_indices)

    return tf.reshape(f_vals, shape=[bs, nv])


def create_one_hot_tensor(indices, max_seq_length):
    """Creates a 3-D one-hot tensor.
    Args:
        indices: An integer tensor of shape [batch_size, num_val].
        max_seq_length: An integer for the last dimension, max value of indices.

    Returns: An 3D one-hot tensor with 1s on positions defined in indices.
    """
    # This returns a tensor of shape [batch_size, num_value, max_seq_len].
    positional_one_hot = tf.one_hot(
        indices, depth=max_seq_length, dtype=tf.float32
    )

    return positional_one_hot


def one_hot_batch_gather(params, indices):
    """Performs batch_gather using one_hot matrix multiplication."""
    batch_size, num_value = modeling.get_shape_list(indices, expected_rank=2)
    _, max_seq_len = modeling.get_shape_list(params, expected_rank=2)

    # This returns a tensor of shape [batch_size, num_value, max_seq_len].
    positional_one_hot = tf.one_hot(
        indices, depth=max_seq_len, dtype=tf.float32
    )

    # Expands the params tensor for selection.
    expand_params = tf.expand_dims(params, 2)

    gathered_values = tf.matmul(positional_one_hot, expand_params)

    # Reduces the last dimension.
    return tf.squeeze(gathered_values, 2)


def compute_span_log_score(start_log_scores, start_pos_list,
                           end_log_scores, end_pos_list, use_gather=False):
    """Computes the span log scores."""
    if use_gather:
        ans_span_start_log_scores = batch_gather(
            start_log_scores, start_pos_list
        )
        ans_span_end_log_scores = batch_gather(
            end_log_scores, end_pos_list
        )
    else:
        ans_span_start_log_scores = one_hot_batch_gather(
            start_log_scores, start_pos_list
        )
        ans_span_end_log_scores = one_hot_batch_gather(
            end_log_scores, end_pos_list
        )

    return (ans_span_start_log_scores + ans_span_end_log_scores)


def compute_logprob(logits, axis=-1, keepdims=None):
    """Computes the log prob based on logits."""
    return logits - tf.reduce_logsumexp(logits, axis=axis, keepdims=keepdims)


def masked_logsumexp(log_score, mask, axis=None, keepdims=True):
    """Computes masked logsumexp score."""
    log_max = tf.stop_gradient(
        tf.reduce_max(log_score, axis=axis, keepdims=keepdims))

    exp_score = tf.exp(log_score - log_max) * mask

    logsumexp = tf.log(
        tf.reduce_sum(exp_score, axis=axis, keepdims=keepdims)) + log_max

    return logsumexp


def doc_nce_loss(start_logits, start_indices, end_logits, end_indices,
                 positions_mask, pos_par_mask, log_rank_score, num_negative,
                 loss_type="mml", use_log_prob=True):
    """Computes document-level normalization span-based losses."""
    # Computes the logigts for a span, which the sum of the corresponding start
    # position logits and the end position logits.

    null_indices = tf.reduce_max(tf.zeros_like(start_indices), axis=-1,
                                 keepdims=True)

    null_start_logits = one_hot_batch_gather(start_logits, null_indices)
    null_end_logits = one_hot_batch_gather(end_logits, null_indices)

    start_logits -= null_start_logits
    end_logits -= null_end_logits

    span_logits = compute_span_log_score(
        start_logits, start_indices, end_logits, end_indices)

    z_psg = (tf.reduce_logsumexp(start_logits, axis=-1)
             + tf.reduce_logsumexp(end_logits, axis=-1))

    log_score_mask = tf.cast(positions_mask, dtype=tf.float32)

    num_neg = tf.cast(num_negative, dtype=tf.float32)

    pos_par_mask = tf.cast(pos_par_mask, dtype=tf.float32)
    if loss_type == "mml":
        p_log_scores = tf.reshape(
            masked_logsumexp(span_logits, log_score_mask, axis=-1), [-1])
        p_positive_psg_log_scores = (
            pos_par_mask * p_log_scores + (1.0 - pos_par_mask) * z_psg)
    else:
        raise ValueError("Unknwon loss_type %s for doc_span_loss!"
                         % loss_type)

    logits = log_rank_score + tf.log(num_neg)

    logits -= p_positive_psg_log_scores

    labels = tf.one_hot(0, depth=(num_negative + 1), dtype=tf.float32)

    nce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                       logits=logits)

    return tf.reduce_mean(nce_loss)


def doc_str_loss(start_logits, start_indices, end_logits, end_indices,
                 positions_mask, pos_par_mask, answer_index_list,
                 max_num_answer_strings,
                 loss_type="positive_par_doc_mml"):
    """Computes document-level normalization string-based losses."""
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, axis=None)
    end_log_prob = compute_logprob(end_logits, axis=None)

    # Computes the log prob for a span, which the sum of the corresponding start
    # position log prob and the end position log prob.
    span_log_prob = compute_span_log_score(
        start_log_prob, start_indices, end_log_prob, end_indices
    )

    max_span_log_prob = tf.stop_gradient(
        tf.reduce_max(span_log_prob, axis=None))

    mask = tf.cast(positions_mask, dtype=tf.float32)
    span_prob = tf.exp(span_log_prob - max_span_log_prob) * mask

    str_prob = group_answer_span_prob(
        span_prob, answer_index_list, max_num_answer_strings
    )

    if loss_type == "doc_mml":
        # The whole document contains one correct string.
        str_loss = -(tf.log(tf.reduce_sum(str_prob, axis=None))
                     + max_span_log_prob)
    elif loss_type == "doc_hard_em":
        str_loss = -(tf.log(tf.reduce_max(str_prob, axis=None))
                     + max_span_log_prob)
    elif loss_type == "all_correct":
        non_zeros = tf.cast(str_prob > 0.0, dtype=tf.float32)
        str_prob = non_zeros * str_prob + (1.0 - non_zeros)
        str_loss = -tf.reduce_sum(
            tf.log(str_prob) + non_zeros * max_span_log_prob, axis=None)
    else:
        raise ValueError("Unknwon loss_type %s for doc_str_loss!"
                         % loss_type)

    return str_loss


def one_hot_answer_positions(position_list, position_mask, depth):
    position_tensor = tf.one_hot(
        position_list, depth, dtype=tf.float32)
    position_masked = position_tensor * tf.cast(
        tf.expand_dims(position_mask, -1), dtype=tf.float32
    )
    onehot_positions = tf.reduce_max(position_masked, axis=1)
    return onehot_positions


def compute_masked_log_score(log_score, position_list, answer_masks, seq_length,
                             log_mask=True, very_neg=-1e3):
    position_tensor = one_hot_answer_positions(
        position_list, answer_masks, seq_length
    )
    if log_mask:
        return log_score + tf.log(position_tensor)
    return log_score + very_neg * (1.0 - position_tensor)


def group_answer_span_prob(ans_span_probs, group_ids, max_num_answer_strings,
                           group_all=True):
    """Sums all answer span probilities from the same group."""
    delta = max_num_answer_strings + 1

    if group_all:
        batch_size = 1
    else:
        batch_size, _ = modeling.get_shape_list(ans_span_probs, expected_rank=2)
        offset = tf.reshape(tf.range(0, batch_size * delta, delta),
                            shape=[batch_size, 1])
        group_ids += offset

    group_ans_probs = tf.math.unsorted_segment_sum(
        ans_span_probs, group_ids, batch_size * delta
    )

    return tf.reshape(
        group_ans_probs, [batch_size, delta]
    )


def compute_max_ans_str_mask(start_logits, start_positions_list, end_logits,
                             end_positions_list, positions_mask,
                             answer_index_list, batch_size, max_num_answers,
                             seq_length, max_num_answer_strings,
                             group_all=True):
    """Computes the max answer string mask."""
    # Here, we assume the full start_logits tensor comes from the same doc.
    start_log_prob = tf.nn.log_softmax(start_logits, axis=None)
    end_log_prob = tf.nn.log_softmax(end_logits, axis=None)

    span_log_prob = compute_span_log_score(
        start_log_prob, start_positions_list,
        end_log_prob, end_positions_list
    )
    span_log_prob -= tf.stop_gradient(tf.reduce_max(span_log_prob, axis=None))
    span_prob = tf.exp(span_log_prob) * tf.cast(positions_mask, dtype=tf.float32)


    str_prob = group_answer_span_prob(
        span_prob, answer_index_list, max_num_answer_strings,
        group_all=group_all
    )

    if group_all:
        max_ans_str_index = tf.tile(
            tf.reshape(tf.argmax(str_prob, axis=-1), shape=[1, 1]),
            [batch_size, max_num_answers]
        )
    else:
        max_ans_str_index = tf.tile(
            tf.reshape(tf.argmax(str_prob, axis=-1), shape=[batch_size, 1]),
            [1, max_num_answers]
        )

    max_ans_str_positions_mask = tf.stop_gradient(tf.cast(tf.equal(
        max_ans_str_index, answer_index_list), tf.int64))

    return max_ans_str_positions_mask


def get_max_mask(log_scores, reduce_all=True):
    """Creates the max value mask for 2D log scores."""
    bs, nv = modeling.get_shape_list(log_scores, expected_rank=2)
    if reduce_all:
        # Only keeps the max log score poistion for the all matrix.
        _, max_indices = tf.nn.top_k(
            tf.reshape(log_scores, [-1]), k=1
        )
        max_mask = tf.reshape(
            tf.one_hot(max_indices, depth=(bs*nv), dtype=tf.float32),
            [bs, nv]
        )
    else:
        # Keeps the max log score poistion for each row of the matrix.
        _, max_indices = tf.nn.top_k(log_scores, k=1)
        max_mask = tf.reduce_max(
            tf.one_hot(max_indices, depth=nv, dtype=tf.float32),
            axis=1
        )

    return tf.stop_gradient(max_mask)


def compute_topk_neg_span_positions(start_logits, end_logits,
                                    pos_start_indices, pos_end_indices,
                                    pos_position_mask,
                                    seq_length,
                                    k=10, span_ub=20):
    """Computes the start and end positions for top-k spans.
    Args:
        start_logits: A float tensor of shape [batch_size, max_seq_length].
        end_logits: A float tensor of shape [batch_size, max_seq_length].
        pos_start_indices: A float tensorf of shape[batch_size, max_num_answers].
        pos_end_indices: A float tensorf of shape[batch_size, max_num_answers].
        seq_length: Integer for the max sequence length.

    Returns:
        start_positions: A integer tensor of shape[batch_size, k].
        end_positions: A integer tensor of shape[batch_size, k].
        valid_span_mask: A float tensor of shape [batch_size, k].
    """
    # Expands the correct positions mask.
    pos_position_mask = tf.cast(
        tf.expand_dims(pos_position_mask, axis=-1), dtype=tf.float32)

    # Computes one-hot matrices for incorrect span start and end positions.
    start_one_hot = 1.0 - tf.reduce_max(
        pos_position_mask * create_one_hot_tensor(
            pos_start_indices, seq_length), axis=1)
    end_one_hot = 1.0 - tf.reduce_max(
        pos_position_mask * create_one_hot_tensor(
            pos_end_indices, seq_length), axis=1)

    bs, _, = modeling.get_shape_list(start_logits, expected_rank=2)

    # First, gets topk start positions of shape [batch_size, k].
    _, start_positions = tf.nn.top_k(start_logits + tf.log(start_one_hot), k=k)

    # Valid end position mask with 0s before start positions and 1s after.
    # The resulting mask tensor is of shape [batch_size, k, seq_length].
    valid_end_pos_mask = 1.0 - tf.sequence_mask(
        start_positions, maxlen=seq_length, dtype=tf.float32)

    # This expanded tensor is of shape [batch_size, 1, seq_length].
    expand_end_logits = tf.tile(
        tf.expand_dims(end_logits + tf.log(end_one_hot), axis=1), [1, k, 1])

    masked_end_logits = expand_end_logits + tf.log(valid_end_pos_mask)

    # The tensor is of shape [batch_size, k, 1]
    _, end_positions = tf.nn.top_k(masked_end_logits, k=1)

    # This will be of shape [batch_size, k].
    end_positions = tf.reshape(end_positions, [bs, k])

    start_before_end = end_positions - start_positions

    # A tensor of shape [batch_size, k], indicating the start-end pair is
    # valid, i.e. start before end and span length is no longer than ub.
    valid_span_mask = tf.stop_gradient(tf.cast(
            start_before_end < span_ub, dtype=tf.float32))

    pos_mask = tf.cast(valid_span_mask, dtype=start_positions.dtype)

    # For invalid spans, the start and end positions should be all 0s.
    start_positions *= pos_mask
    end_positions *= pos_mask

    return start_positions, end_positions, valid_span_mask


def compute_span_embeddings(pos_hiddens, start_indices, end_indices, seq_length,
                            return_span=True):
    """Computes the span embeddings based on the start and end positions."""
    # Computes one-hot matrices of shape [batch_size, num_pos, max_seq_length].
    start_one_hot = create_one_hot_tensor(start_indices, seq_length)
    end_one_hot = create_one_hot_tensor(end_indices, seq_length)

    # Both hiddens of shape [batch_size, num_pos, hidden_dim].
    start_hidden = tf.matmul(start_one_hot, pos_hiddens)
    end_hidden = tf.matmul(end_one_hot, pos_hiddens)

    if not return_span:
        return start_hidden, end_hidden

    span_hidden = tf.concat([start_hidden, end_hidden], axis=-1)

    return span_hidden


def compute_span_embeddings_v2(pos_hiddens, start_indices, end_indices, seq_length):
    """Computes the span embeddings based on the start and end positions."""
    # Computes one-hot matrices of shape [batch_size, num_pos, max_seq_length].
    start_one_hot = create_one_hot_tensor(start_indices, seq_length)
    end_one_hot = create_one_hot_tensor(end_indices, seq_length)

    # Both hiddens of shape [batch_size, num_pos, hidden_dim].
    start_hidden = tf.matmul(start_one_hot, pos_hiddens)
    end_hidden = tf.matmul(end_one_hot, pos_hiddens)

    span_hidden = tf.concat([start_hidden, end_hidden], axis=-1)

    return span_hidden, start_hidden, end_hidden


def l2_normalize_vector(d, epsilon1=1e-12, epsilon2=1e-6):
  """Normalizes vector."""
  d /= tf.stop_gradient(epsilon1 + tf.reduce_max(tf.abs(d), axis=-1, keep_dims=True))
  d /= tf.stop_gradient(
      tf.sqrt(epsilon2 + tf.reduce_sum(tf.square(d), axis=-1, keep_dims=True)))
  return d


def contrastive_projection(x, output_dim, initializer_range=0.02,
                           act_func="relu", scope="contrastive_layer",
                           reuse=None):
    """Computes a constrastive projection of input."""
    with tf.variable_scope(scope or "contrastive_layer", reuse=reuse):
        y = tf.layers.dense(
            x,
            output_dim,
            kernel_initializer=modeling.create_initializer(initializer_range),
            activation=modeling.get_activation(act_func),
            name="non_linear_dense",
        )
        y = tf.layers.dense(
            y,
            output_dim,
            kernel_initializer=modeling.create_initializer(initializer_range),
            activation=None,
            use_bias=False,
            name="linear_dense",
        )

    return y


def doc_contrastive_pos_loss(model, start_logits, end_logits, pos_start_indices,
                             pos_end_indices, answer_positions_mask, temperature=1.0,
                             project_dim=768, neg_k=5, span_ub=20):
    """Computes document-level constrastive losses."""
    final_hidden = model.get_sequence_output()
    bs, seq_length, hidden_dim = modeling.get_shape_list(
        final_hidden, expected_rank=3)

    (neg_start_indices, neg_end_indices,
     neg_positions_mask) = compute_topk_neg_span_positions(
         start_logits, end_logits, pos_start_indices, pos_end_indices,
         answer_positions_mask,
         seq_length, k=neg_k, span_ub=span_ub)

    pos_par_mask = tf.reduce_max(answer_positions_mask, axis=-1, keepdims=True)

    first_token_pos = tf.ones_like(pos_par_mask)

    question_reps, _ = compute_span_embeddings(
        final_hidden, first_token_pos, first_token_pos, seq_length,
        return_span=False)

    pos_par_mask = tf.cast(pos_par_mask, dtype=tf.float32)
    question_reps = tf.reduce_sum(
        tf.expand_dims(pos_par_mask, axis=-1) * question_reps,
        axis=0, keepdims=True) / tf.reduce_sum(pos_par_mask)

    question_reps = contrastive_projection(
        question_reps, project_dim, reuse=False, scope="q_contrast_proj")

    # Tensor of shape [batch_size, num_span, hidden_dim * 2].
    pos_start_reps, pos_end_reps = compute_span_embeddings(
        final_hidden, pos_start_indices, pos_end_indices, seq_length,
        return_span=False)
    # pos_start_reps = contrastive_projection(
    #     pos_start_reps, project_dim, reuse=False, scope="start_contrast_proj")
    # pos_end_reps = contrastive_projection(
    #     pos_end_reps, project_dim, reuse=False, scope="end_contrast_proj")
    pos_start_reps = contrastive_projection(
        pos_start_reps, project_dim, reuse=False, scope="pos_contrast_proj")
    pos_end_reps = contrastive_projection(
        pos_end_reps, project_dim, reuse=True, scope="pos_contrast_proj")

    neg_start_reps, neg_end_reps = compute_span_embeddings(
        final_hidden, neg_start_indices, neg_end_indices, seq_length,
        return_span=False)
    # neg_start_reps = contrastive_projection(
    #     neg_start_reps, project_dim, reuse=True, scope="start_contrast_proj")
    # neg_end_reps = contrastive_projection(
    #     neg_end_reps, project_dim, reuse=True, scope="end_contrast_proj")
    neg_start_reps = contrastive_projection(
        neg_start_reps, project_dim, reuse=True, scope="pos_contrast_proj")
    neg_end_reps = contrastive_projection(
        neg_end_reps, project_dim, reuse=True, scope="pos_contrast_proj")

    # Normalizes all embeddings to the L2 unit.
    # Tensor of shape [batch_size, 1, hidden_dim].
    norm_question_reps = l2_normalize_vector(
        tf.tile(question_reps, [bs, 1, 1]))

    # Both of shape [batch_size, num_spans, hidden_dim].
    norm_pos_start_reps = l2_normalize_vector(pos_start_reps)
    norm_pos_end_reps = l2_normalize_vector(pos_end_reps)

    norm_neg_start_reps = l2_normalize_vector(neg_start_reps)
    norm_neg_end_reps = l2_normalize_vector(neg_end_reps)

    # Scores of shape [batch_size, num_span].
    pos_start_sim_scores = tf.reduce_max(
        tf.matmul(norm_pos_start_reps, norm_question_reps, transpose_b=True),
        axis=-1)
    pos_end_sim_scores = tf.reduce_max(
        tf.matmul(norm_pos_end_reps, norm_question_reps, transpose_b=True),
        axis=-1)

    # Scores of shape [batch_size, num_span].
    neg_start_sim_scores = temperature * tf.reduce_max(
        tf.matmul(norm_neg_start_reps, norm_question_reps, transpose_b=True),
        axis=-1)
    neg_end_sim_scores = temperature * tf.reduce_max(
        tf.matmul(norm_neg_end_reps, norm_question_reps, transpose_b=True),
        axis=-1)

    answer_positions_mask = tf.cast(answer_positions_mask, dtype=tf.float32)
    num_pos_spans = tf.reduce_sum(answer_positions_mask)
    pos_start_loss = temperature * tf.reduce_sum(
        pos_start_sim_scores * answer_positions_mask) / num_pos_spans
    pos_end_loss = temperature * tf.reduce_sum(
        pos_end_sim_scores * answer_positions_mask) / num_pos_spans

    neg_positions_mask = tf.cast(neg_positions_mask, dtype=tf.float32)
    neg_start_loss = tf.reduce_logsumexp(
        neg_start_sim_scores + tf.log(neg_positions_mask)
    )
    neg_end_loss = tf.reduce_logsumexp(
        neg_end_sim_scores + tf.log(neg_positions_mask)
    )

    contrastive_loss = 0.5 * (
        2.0 - pos_start_loss - pos_end_loss + neg_start_loss + neg_end_loss)
    return contrastive_loss


def doc_contrastive_loss_v2(model, start_logits, end_logits, pos_start_indices,
                            pos_end_indices, answer_positions_mask, temperature=1.0,
                            project_dim=768, neg_k=5, span_ub=20):
    """Computes document-level constrastive losses."""
    final_hidden = model.get_sequence_output()
    bs, seq_length, hidden_dim = modeling.get_shape_list(
        final_hidden, expected_rank=3)

    (neg_start_indices, neg_end_indices,
     neg_positions_mask) = compute_topk_neg_span_positions(
         start_logits, end_logits, pos_start_indices, pos_end_indices,
         answer_positions_mask,
         seq_length, k=neg_k, span_ub=span_ub)

    pos_par_mask = tf.reduce_max(answer_positions_mask, axis=-1, keepdims=True)

    first_token_pos = tf.ones_like(pos_par_mask)

    question_reps, _ = compute_span_embeddings(
        final_hidden, first_token_pos, first_token_pos, seq_length,
        return_span=False)

    pos_par_mask = tf.cast(pos_par_mask, dtype=tf.float32)
    question_reps = tf.reduce_sum(
        tf.expand_dims(pos_par_mask, axis=-1) * question_reps,
        axis=0, keepdims=True) / tf.reduce_sum(pos_par_mask)

    question_reps = contrastive_projection(
        question_reps, project_dim, reuse=False, scope="q_contrast_proj")

    # Tensor of shape [batch_size, num_span, hidden_dim * 2].
    pos_span_reps, pos_start_reps, pos_end_reps = compute_span_embeddings_v2(
        final_hidden, pos_start_indices, pos_end_indices, seq_length)
    pos_span_reps = contrastive_projection(
        pos_span_reps, project_dim, reuse=False, scope="span_contrast_proj")

    # Positional representations.
    pos_start_reps = contrastive_projection(
        pos_start_reps, project_dim, reuse=False, scope="pos_contrast_proj")
    pos_end_reps = contrastive_projection(
        pos_end_reps, project_dim, reuse=True, scope="pos_contrast_proj")

    neg_span_reps, neg_start_reps, neg_end_reps = compute_span_embeddings_v2(
        final_hidden, neg_start_indices, neg_end_indices, seq_length
    )
    neg_span_reps = contrastive_projection(
        neg_span_reps, project_dim, reuse=True, scope="span_contrast_proj")

    neg_start_reps = contrastive_projection(
        neg_start_reps, project_dim, reuse=True, scope="pos_contrast_proj")
    neg_end_reps = contrastive_projection(
        neg_end_reps, project_dim, reuse=True, scope="pos_contrast_proj")

    # Normalizes all embeddings to the L2 unit.
    # Tensor of shape [batch_size, 1, hidden_dim].
    norm_question_reps = l2_normalize_vector(
        tf.tile(question_reps, [bs, 1, 1]))

    # Both of shape [batch_size, num_spans, hidden_dim].
    norm_pos_span_reps = l2_normalize_vector(pos_span_reps)
    norm_neg_span_reps = l2_normalize_vector(neg_span_reps)

    # Positional representations.
    norm_pos_start_reps = l2_normalize_vector(pos_start_reps)
    norm_pos_end_reps = l2_normalize_vector(pos_end_reps)

    norm_neg_start_reps = l2_normalize_vector(neg_start_reps)
    norm_neg_end_reps = l2_normalize_vector(neg_end_reps)

    # Scores of shape [batch_size, num_span].
    pos_sim_scores = tf.reduce_max(
        tf.matmul(norm_pos_span_reps, norm_question_reps, transpose_b=True),
        axis=-1)

    # Scores of shape [batch_size, num_span].
    neg_sim_scores = temperature * tf.reduce_max(
        tf.matmul(norm_neg_span_reps, norm_question_reps, transpose_b=True),
        axis=-1)

    answer_positions_mask = tf.cast(answer_positions_mask, dtype=tf.float32)
    num_pos_spans = tf.reduce_sum(answer_positions_mask)
    pos_loss = temperature * tf.reduce_sum(
        pos_sim_scores * answer_positions_mask) / num_pos_spans

    neg_positions_mask = tf.cast(neg_positions_mask, dtype=tf.float32)
    neg_loss = tf.reduce_logsumexp(
        neg_sim_scores + tf.log(neg_positions_mask)
    )

    # Matched start and end position representations are regularized.
    p_pos_sim_scores = temperature * tf.reduce_sum(
        norm_pos_start_reps * norm_pos_end_reps) / num_pos_spans

    contrastive_loss = 2.0 - pos_loss + neg_loss - p_pos_sim_scores
    return contrastive_loss



def doc_contrastive_loss(model, start_logits, end_logits, pos_start_indices,
                         pos_end_indices, answer_positions_mask, temperature=1.0,
                         project_dim=768, neg_k=5, span_ub=20):
    """Computes document-level constrastive losses."""
    final_hidden = model.get_sequence_output()
    bs, seq_length, hidden_dim = modeling.get_shape_list(
        final_hidden, expected_rank=3)

    (neg_start_indices, neg_end_indices,
     neg_positions_mask) = compute_topk_neg_span_positions(
         start_logits, end_logits, pos_start_indices, pos_end_indices,
         answer_positions_mask,
         seq_length, k=neg_k, span_ub=span_ub)

    pos_par_mask = tf.reduce_max(answer_positions_mask, axis=-1, keepdims=True)

    first_token_pos = tf.ones_like(pos_par_mask)

    question_reps, _ = compute_span_embeddings(
        final_hidden, first_token_pos, first_token_pos, seq_length,
        return_span=False)

    pos_par_mask = tf.cast(pos_par_mask, dtype=tf.float32)
    question_reps = tf.reduce_sum(
        tf.expand_dims(pos_par_mask, axis=-1) * question_reps,
        axis=0, keepdims=True) / tf.reduce_sum(pos_par_mask)

    question_reps = contrastive_projection(
        question_reps, project_dim, reuse=False, scope="q_contrast_proj")

    # Tensor of shape [batch_size, num_span, hidden_dim * 2].
    pos_span_reps = compute_span_embeddings(
        final_hidden, pos_start_indices, pos_end_indices, seq_length)
    pos_span_reps = contrastive_projection(
        pos_span_reps, project_dim, reuse=False, scope="span_contrast_proj")

    neg_span_reps = compute_span_embeddings(
        final_hidden, neg_start_indices, neg_end_indices, seq_length
    )
    neg_span_reps = contrastive_projection(
        neg_span_reps, project_dim, reuse=True, scope="span_contrast_proj")

    # Normalizes all embeddings to the L2 unit.
    # Tensor of shape [batch_size, 1, hidden_dim].
    norm_question_reps = l2_normalize_vector(
        tf.tile(question_reps, [bs, 1, 1]))

    # Both of shape [batch_size, num_spans, hidden_dim].
    norm_pos_span_reps = l2_normalize_vector(pos_span_reps)
    norm_neg_span_reps = l2_normalize_vector(neg_span_reps)

    # Scores of shape [batch_size, num_span].
    pos_sim_scores = tf.reduce_max(
        tf.matmul(norm_pos_span_reps, norm_question_reps, transpose_b=True),
        axis=-1)

    # Scores of shape [batch_size, num_span].
    neg_sim_scores = temperature * tf.reduce_max(
        tf.matmul(norm_neg_span_reps, norm_question_reps, transpose_b=True),
        axis=-1)

    answer_positions_mask = tf.cast(answer_positions_mask, dtype=tf.float32)
    num_pos_spans = tf.reduce_sum(answer_positions_mask)
    pos_loss = temperature * tf.reduce_sum(
        pos_sim_scores * answer_positions_mask) / num_pos_spans

    neg_positions_mask = tf.cast(neg_positions_mask, dtype=tf.float32)
    neg_loss = tf.reduce_logsumexp(
        neg_sim_scores + tf.log(neg_positions_mask)
    )

    contrastive_loss = 1.0 - pos_loss + neg_loss
    return contrastive_loss


def doc_pos_loss(start_logits, start_indices, end_logits, end_indices,
                 answer_positions_mask, pos_par_mask, seq_length,
                 logit_temp=1.0, loss_type="positive_par_doc_mml", pd_loss=None):
    """Computes document-level normalization position-based losses."""
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, axis=None)
    end_log_prob = compute_logprob(end_logits, axis=None)

    masked_start_log_prob = compute_masked_log_score(
        start_log_prob, start_indices, answer_positions_mask, seq_length,
        log_mask=False if pd_loss and pd_loss == "js" else True
    )
    masked_end_log_prob = compute_masked_log_score(
        end_log_prob, end_indices, answer_positions_mask, seq_length,
        log_mask=False if pd_loss and pd_loss == "js" else True
    )

    if 'hard_em' in loss_type:
        # Computes the log prob for a span, which the sum of the corresponding start
        # position log prob and the end position log prob.
        span_log_prob = compute_span_log_score(
            start_log_prob, start_indices, end_log_prob, end_indices
        ) + tf.log(tf.cast(answer_positions_mask, dtype=tf.float32))

    if loss_type == "positive_par_doc_mml":
        # Each positive paragraph contains a correct span.
        pos_par_mask = tf.cast(pos_par_mask, dtype=tf.float32)
        start_loss = tf.reduce_sum(
            -pos_par_mask * tf.reduce_logsumexp(masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_sum(
            -pos_par_mask * tf.reduce_logsumexp(masked_end_log_prob, axis=-1))
    elif loss_type == "positive_par_doc_hard_em":
        max_span_mask = get_max_mask(span_log_prob, reduce_all=False)
        max_masked_start_log_prob = compute_masked_log_score(
            start_log_prob, start_indices, max_span_mask, seq_length
        )
        max_masked_end_log_prob = compute_masked_log_score(
            end_log_prob, end_indices, max_span_mask, seq_length
        )

        # Each positive paragraph contains a correct span.
        pos_par_mask = tf.cast(pos_par_mask, dtype=tf.float32)
        start_loss = tf.reduce_sum(
            -pos_par_mask * tf.reduce_max(max_masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_sum(
            -pos_par_mask * tf.reduce_max(max_masked_end_log_prob, axis=-1))
    elif loss_type == "doc_mml":
        # The whole document contains one correct span.
        # start_loss = -tf.reduce_logsumexp(masked_start_log_prob, axis=None)
        # end_loss = -tf.reduce_logsumexp(masked_end_log_prob, axis=None)

        masked_start_logits = compute_masked_log_score(
            start_logits, start_indices, answer_positions_mask, seq_length
        )
        masked_end_logits = compute_masked_log_score(
            end_logits, end_indices, answer_positions_mask, seq_length
        )
        z_start = tf.reduce_logsumexp(start_logits, axis=None)
        z_end = tf.reduce_logsumexp(end_logits, axis=None)

        start_scores = tf.reduce_logsumexp(masked_start_logits, axis=None)
        end_scores = tf.reduce_logsumexp(masked_end_logits, axis=None)

        start_loss = z_start - start_scores
        end_loss = z_end - end_scores
    elif loss_type == "doc_hard_em":
        max_span_mask = get_max_mask(span_log_prob, reduce_all=True)
        max_masked_start_log_prob = compute_masked_log_score(
            start_log_prob, start_indices, max_span_mask, seq_length
        )
        max_masked_end_log_prob = compute_masked_log_score(
            end_log_prob, end_indices, max_span_mask, seq_length
        )

        start_loss = -tf.reduce_max(
            max_masked_start_log_prob, axis=None)
        end_loss = -tf.reduce_max(
            max_masked_end_log_prob, axis=None)
    elif loss_type == "all_correct":
        # The whole document contains one correct span.
        start_position_tensor = one_hot_answer_positions(
            start_indices, answer_positions_mask, seq_length
        )
        end_position_tensor = one_hot_answer_positions(
            end_indices, answer_positions_mask, seq_length
        )

        start_loss = tf.reduce_sum(-start_position_tensor * start_log_prob, axis=None)
        end_loss = tf.reduce_sum(-end_position_tensor * end_log_prob, axis=None)
    elif loss_type == "pd":
        if pd_loss is None:
            raise ValueError("When using pd loss, pd_loss must be defined!")

        if pd_loss == "v3":
            tf.logging.info("Using KL for pd loss.")
            masked_start_p = tf.stop_gradient(tf.exp(masked_start_log_prob))
            masked_end_p = tf.stop_gradient(tf.exp(masked_end_log_prob))
            start_loss = tf.reduce_sum(-masked_start_p * start_log_prob)
            end_loss = tf.reduce_sum(-masked_end_p * end_log_prob)
        elif pd_loss == "hellinger":
            tf.logging.info("Using hellinger for pd loss.")
            masked_start_p = tf.stop_gradient(
                tf.exp(0.5 * masked_start_log_prob))
            masked_end_p = tf.stop_gradient(
                tf.exp(0.5 * masked_end_log_prob))
            start_p = tf.exp(start_log_prob)
            end_p = tf.exp(end_log_prob)
            start_loss = tf.reduce_sum(tf.square(masked_start_p - start_p))
            end_loss = tf.reduce_sum(tf.square(masked_end_p - end_p))
        elif pd_loss == "js":
            tf.logging.info("Using js for pd loss.")
            masked_start_p = tf.stop_gradient(
                tf.exp(masked_start_log_prob))
            masked_end_p = tf.stop_gradient(
                tf.exp(masked_end_log_prob))
            masked_start_log_prob = tf.stop_gradient(masked_start_log_prob)
            masked_end_log_prob = tf.stop_gradient(masked_end_log_prob)
            start_p = tf.exp(start_log_prob)
            end_p = tf.exp(end_log_prob)

            mean_start_logp = tf.reduce_logsumexp(
                tf.concat([
                    tf.expand_dims(start_log_prob, axis=-1),
                    tf.expand_dims(masked_start_log_prob, axis=-1)
                ], axis=-1), axis=-1) + tf.log(0.5)
            mean_end_logp = tf.reduce_logsumexp(
                tf.concat([
                    tf.expand_dims(end_log_prob, axis=-1),
                    tf.expand_dims(masked_end_log_prob, axis=-1)
                ], axis=-1), axis=-1) + tf.log(0.5)

            start_loss = 0.5 * tf.reduce_sum(
                start_p * start_log_prob +
                masked_start_p * masked_start_log_prob -
                (start_p + masked_start_p) * mean_start_logp)
            end_loss = 0.5 * tf.reduce_sum(
                end_p * end_log_prob +
                masked_end_p * masked_end_log_prob -
                (end_p + masked_end_p) * mean_end_logp)
        else:
            raise ValueError("Unknown pd_loss %s" % pd_loss)

    else:
        raise ValueError("Unknwon loss_type %s for doc_pos_loss!"
                         % loss_type)

    return (start_loss + end_loss) / 2.0


def doc_span_loss(start_logits, start_indices, end_logits, end_indices,
                  positions_mask, pos_par_mask, loss_type="doc_mml"):
    """Computes doc-level normalization span-based losses."""
    # Computes the log prob for start and end positions.
    # start_log_prob = compute_logprob(start_logits, axis=-1, keepdims=True)
    # end_log_prob = compute_logprob(end_logits, axis=-1, keepdims=True)

    # Computes the log prob for a span, which the sum of the corresponding start
    # position log prob and the end position log prob.
    span_logits = compute_span_log_score(
        start_logits, start_indices, end_logits, end_indices)

    masked_span_logits = span_logits + tf.log(
        tf.cast(positions_mask, dtype=tf.float32))

    z_all = (tf.reduce_logsumexp(start_logits, axis=None)
             + tf.reduce_logsumexp(end_logits, axis=None))

    if loss_type == "doc_mml":
        span_loss = z_all - tf.reduce_logsumexp(masked_span_logits)
    elif loss_type == "doc_hard_em":
        span_loss = z_all - tf.reduce_max(masked_span_logits)
    elif loss_type == "positive_par_doc_mml":
        # Each positive paragraph contains a correct span.
        pos_par_mask = tf.cast(pos_par_mask, dtype=tf.float32)
        span_loss = tf.reduce_sum(pos_par_mask * (
            z_all - tf.reduce_logsumexp(masked_span_logits, axis=-1)))
    elif loss_type == "positive_par_doc_hard_em":
        pos_par_mask = tf.cast(pos_par_mask, dtype=tf.float32)
        span_loss = tf.reduce_sum(pos_par_mask * (
            z_all - tf.reduce_max(masked_span_logits, axis=-1)))
    else:
        raise ValueError("Unknwon loss_type %s for doc_span_loss!"
                         % loss_type)

    return span_loss


def par_span_loss(start_logits, start_indices, end_logits, end_indices,
                  positions_mask, loss_type="par_mml"):
    """Computes paragraph-level normalization span-based losses."""
    # Computes the log prob for start and end positions.
    # start_log_prob = compute_logprob(start_logits, axis=-1, keepdims=True)
    # end_log_prob = compute_logprob(end_logits, axis=-1, keepdims=True)

    # Computes the log prob for a span, which the sum of the corresponding start
    # position log prob and the end position log prob.
    span_log_logits = compute_span_log_score(
        start_logits, start_indices, end_logits, end_indices
    )

    z_all = (tf.reduce_logsumexp(start_logits, axis=-1)
             + tf.reduce_logsumexp(end_logits, axis=-1))

    masked_span_logits = span_log_logits + tf.log(
        tf.cast(positions_mask, dtype=tf.float32)
    )

    if loss_type == "par_mml":
        # Each positive paragraph contains a correct span.
        span_loss = tf.reduce_mean(
            z_all - tf.reduce_logsumexp(masked_span_logits, axis=-1))
    elif loss_type == "par_hard_em":
        span_loss = tf.reduce_mean(
            z_all - tf.reduce_max(masked_span_logits, axis=-1))
    else:
        raise ValueError("Unknwon loss_type %s for par_span_loss!"
                         % loss_type)

    return span_loss


def answer_pos_loss(start_logits, start_indices, end_logits, end_indices,
                    answer_positions_mask, seq_length, num_doc_features=None,
                    loss_type="par_mml"):
    """Computes position-based losses."""
    # Computes the log prob for start and end positions.
    # start_log_prob = compute_logprob(start_logits, axis=-1, keepdims=True)
    # end_log_prob = compute_logprob(end_logits, axis=-1, keepdims=True)

    # masked_start_log_prob = compute_masked_log_score(
    #     start_log_prob, start_indices, answer_positions_mask, seq_length
    # )

    # masked_end_log_prob = compute_masked_log_score(
    #     end_log_prob, end_indices, answer_positions_mask, seq_length
    # )

    z_start = tf.reduce_logsumexp(start_logits, axis=-1)
    z_end = tf.reduce_logsumexp(end_logits, axis=-1)

    masked_start_logits = compute_masked_log_score(
        start_logits, start_indices, answer_positions_mask, seq_length
    )
    masked_end_logits = compute_masked_log_score(
        end_logits, end_indices, answer_positions_mask, seq_length
    )

    if 'hard_em' in loss_type:
        # Computes the log prob for a span, which the sum of the corresponding start
        # position log prob and the end position log prob.
        span_log_prob = compute_span_log_score(
            # start_log_prob, start_indices, end_log_prob, end_indices
            start_logits, start_indices, end_logits, end_indices
        ) + tf.log(tf.cast(answer_positions_mask, dtype=tf.float32))

    if loss_type == "par_mml":
        # Each positive paragraph contains a correct span.
        # start_loss = tf.reduce_mean(
        #     -tf.reduce_logsumexp(masked_start_log_prob, axis=-1))
        # end_loss = tf.reduce_mean(
        #     -tf.reduce_logsumexp(masked_end_log_prob, axis=-1))

        mml_start_log_scores = tf.reduce_logsumexp(masked_start_logits, axis=-1)
        mml_end_log_scores = tf.reduce_logsumexp(masked_end_logits, axis=-1)

        start_loss = tf.reduce_mean(z_start - mml_start_log_scores)
        end_loss = tf.reduce_mean(z_end - mml_end_log_scores)
    elif loss_type == "par_hard_em":
        max_span_mask = get_max_mask(span_log_prob, reduce_all=False)
        # max_masked_start_log_prob = compute_masked_log_score(
        #     start_log_prob, start_indices, max_span_mask, seq_length
        # )
        # max_masked_end_log_prob = compute_masked_log_score(
        #     end_log_prob, end_indices, max_span_mask, seq_length
        # )
        max_masked_start_logits = compute_masked_log_score(
            start_logits, start_indices, max_span_mask, seq_length
        )
        max_masked_end_logits = compute_masked_log_score(
            end_logits, end_indices, max_span_mask, seq_length
        )

        start_loss = tf.reduce_mean(
            z_start - tf.reduce_max(max_masked_start_logits, axis=-1))
        end_loss = tf.reduce_mean(
            z_end - tf.reduce_max(max_masked_end_logits, axis=-1))
    else:
        raise ValueError("Unknwon loss_type %s for par_pos_loss!"
                         % loss_type)

    return (start_loss + end_loss) / 2.0


def par_pos_loss(start_logits, start_indices, end_logits, end_indices,
                 answer_positions_mask, seq_length, loss_type="par_mml",
                 pd_loss=None, logit_temp=1.0):
    """Computes document-level normalization position-based losses."""
    # Computes the log prob for start and end positions.
    start_log_prob = compute_logprob(start_logits, axis=-1, keepdims=True)
    end_log_prob = compute_logprob(end_logits, axis=-1, keepdims=True)

    masked_start_log_prob = compute_masked_log_score(
        start_log_prob, start_indices, answer_positions_mask, seq_length
    )

    masked_end_log_prob = compute_masked_log_score(
        end_log_prob, end_indices, answer_positions_mask, seq_length
    )

    if 'hard_em' in loss_type:
        # Computes the log prob for a span, which the sum of the corresponding start
        # position log prob and the end position log prob.
        span_log_prob = compute_span_log_score(
            start_log_prob, start_indices, end_log_prob, end_indices
        ) + tf.log(tf.cast(answer_positions_mask, dtype=tf.float32))

    if loss_type == "par_mml":
        # Each positive paragraph contains a correct span.
        start_loss = tf.reduce_mean(
            -tf.reduce_logsumexp(masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_mean(
            -tf.reduce_logsumexp(masked_end_log_prob, axis=-1))
    elif loss_type == "par_hard_em":
        max_span_mask = get_max_mask(span_log_prob, reduce_all=False)
        max_masked_start_log_prob = compute_masked_log_score(
            start_log_prob, start_indices, max_span_mask, seq_length
        )
        max_masked_end_log_prob = compute_masked_log_score(
            end_log_prob, end_indices, max_span_mask, seq_length
        )

        start_loss = tf.reduce_mean(
            -tf.reduce_max(max_masked_start_log_prob, axis=-1))
        end_loss = tf.reduce_mean(
            -tf.reduce_max(max_masked_end_log_prob, axis=-1))
    elif loss_type == "pd":
        if pd_loss is None:
            raise ValueError("When using pd loss, pd_loss must be defined!")
        if logit_temp < 1.0:
            # Performs dampening on the logits.
            tf.logging.info("Dampening the logits with temperature %s" %
                            logit_temp)
            start_logits *= logit_temp
            end_logits *= logit_temp

        masked_start_logp = tf.stop_gradient(tf.nn.log_softmax(compute_masked_log_score(
            start_logits, start_indices, answer_positions_mask, seq_length
        )))
        masked_end_logp = tf.stop_gradient(tf.nn.log_softmax(compute_masked_log_score(
            end_logits, end_indices, answer_positions_mask, seq_length
        )))
        start_logp = tf.nn.log_softmax(start_logits)
        end_logp = tf.nn.log_softmax(end_logits)

        start_loss = tf.reduce_sum(pd_loss(masked_start_logp, start_logp))
        end_loss = tf.reduce_sum(pd_loss(masked_end_logp, end_logp))

    else:
        raise ValueError("Unknwon loss_type %s for par_pos_loss!"
                         % loss_type)

    return (start_loss + end_loss) / 2.0

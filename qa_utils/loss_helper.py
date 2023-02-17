"""This script contains helper loss functions for QA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import modeling


LARGE_VAL = 1e6


def one_hot_answer_positions(position_list, position_mask, depth):
    position_tensor = tf.one_hot(
        position_list, depth, dtype=tf.float32)
    position_masked = position_tensor * tf.cast(
        tf.expand_dims(position_mask, -1), dtype=tf.float32
    )
    onehot_positions = tf.reduce_max(position_masked, axis=1)
    return onehot_positions


def margin_func(t, lb, ub):
    """Computes min(max(lb, t), ub)."""
    def hs_func(x, s):
        """Computes max(0, s - x)."""
        return tf.nn.relu(s - x)

    return hs_func(t, lb) - hs_func(t, ub) + ub


def compute_margin_loss(logits, one_hot_positions, tau=24, epsilon=1e-5):
    """This computes the margin loss for the correctly classifiy case."""
    label_logits = tf.reduce_max(
        tf.log(one_hot_positions) + logits, axis=-1
    )
    best_wrong_logits = tf.reduce_max(
        tf.log(1.0 - one_hot_positions) + logits, axis=-1
    )
    return -tf.reduce_sum(margin_func(
        label_logits - best_wrong_logits, epsilon, tau))

def compute_pos_margin_loss(logits, one_hot_positions, tau=24, epsilon=1e-5):
    """This computes the margin loss for the correctly classifiy case."""
    label_logits = one_hot_positions * logits + (one_hot_positions - 1.0) * logits
    return -tf.reduce_sum(margin_func(
        label_logits, epsilon, tau))


def kl_divergence(p, log_p, log_q, axis=-1):
    """Computes KL divergence. BP through Q."""
    # p = tf.stop_gradient(p)
    # log_p = tf.stop_gradient(log_p)
    return tf.reduce_sum(p * (log_p - log_q), axis=axis)


def kl_divergence_w_log_prob(log_p, log_q, axis=-1):
    """Computes KL divergence. BP through Q."""
    # p = tf.stop_gradient(p)
    # log_p = tf.stop_gradient(log_p)
    return tf.reduce_sum(tf.exp(log_p) * (log_p - log_q), axis=axis)


def hellinger_distance_w_log_prob(log_p, log_q, axis=-1):
    """Computes squared hellinger distance."""
    return 0.5 * tf.reduce_sum(tf.square(
        tf.exp(log_p / 2.0) - tf.exp(log_q / 2.0)), axis=axis)


def js_divergence_w_log_prob(log_p, log_q, axis=-1):
    """Computes js divergence."""
    # mean_log_prob = tf.log(0.5 * (tf.exp(log_p) + tf.exp(log_q)))
    mean_log_prob = tf.stop_gradient(tf.log(0.5) + tf.reduce_logsumexp(
        tf.concat([tf.expand_dims(log_p, axis=-1),
                   tf.expand_dims(log_q, axis=-1)],
                  axis=-1), axis=-1))
    return (
        0.5 * kl_divergence_w_log_prob(log_p, mean_log_prob, axis=axis) +
        0.5 * kl_divergence_w_log_prob(log_q, mean_log_prob, axis=axis)
    )


def rev_kl_divergence(p, log_p, log_q, axis=-1):
    """Computes reverse-KL divergence. BP through P."""
    log_q = tf.stop_gradient(log_q)
    # log_p = tf.stop_gradient(log_p)
    return tf.reduce_sum(p * (log_p - log_q), axis=axis)


def kl_divergence_w_logits(
    logits_p, logits_q, label_weights=None, axis=-1, epsilon=1e-6,
    reverse_kl=False,
):
    """Computes KL divergence."""
    logits_p = tf.cast(logits_p, dtype=tf.float32)
    logits_q = tf.cast(logits_q, dtype=tf.float32)

    log_p = tf.nn.log_softmax(logits_p, axis=axis)
    log_q = tf.nn.log_softmax(logits_q, axis=axis)

    if reverse_kl:
        q = tf.nn.softmax(logits_q, axis=-1)
        kl_div = rev_kl_divergence(q, log_q, log_p, axis=axis)
    else:
        p = tf.nn.softmax(logits_p, axis=-1)
        kl_div = kl_divergence(p, log_p, log_q, axis=axis)

    if label_weights is not None:
        label_weights = tf.reshape(label_weights, [-1])
        num = tf.reduce_sum(kl_div * label_weights)
        dem = tf.reduce_sum(label_weights) + epsilon
        kl_div = num / dem
    else:
        kl_div = tf.reduce_sum(kl_div)

    return kl_div


def symmetric_kl(logits_p, logits_q, label_weights=None, axis=-1, epsilon=1e-6):
    """Computes symmetric KL."""
    logits_p = tf.cast(logits_p, dtype=tf.float32)
    logits_q = tf.cast(logits_q, dtype=tf.float32)

    p = tf.nn.softmax(logits_p, axis=-1)
    log_p = tf.nn.log_softmax(logits_p, axis=-1)
    q = tf.nn.softmax(logits_q, axis=-1)
    log_q = tf.nn.log_softmax(logits_q, axis=-1)

    per_example_loss = 0.5 * (
        kl_divergence(p, log_p, log_q, axis=axis)
        + rev_kl_divergence(q, log_q, log_p, axis=axis)
    )

    if label_weights is not None:
        label_weights = tf.reshape(label_weights, [-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + epsilon
        loss = numerator / denominator
    else:
        loss = tf.reduce_mean(per_example_loss)

    return loss


def compute_forward_logits(final_hidden, output_var_scope="cls"):
    """Computes the logits."""
    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    with tf.variable_scope(output_var_scope, reuse=True):
        output_weights = tf.get_variable(
            "squad/output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "squad/output_bias", [2], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden,
                                         [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [batch_size, seq_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])

        unstacked_logits = tf.unstack(logits, axis=0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    # start_logits = tf.cast(start_logits, dtype=tf.float32)
    # end_logits = tf.cast(end_logits, dtype=tf.float32)
    return start_logits, end_logits


def compute_double_forward_loss(start_logits, end_logits, model, output_var_scope="cls"):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(model.get_embedding_output())

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    double_forward_loss = (
        symmetric_kl(start_logits, d_start_logits) +
        symmetric_kl(end_logits, d_end_logits)) / 2.0

    return double_forward_loss


def compute_double_forward_loss_v2(start_logits, end_logits, model, output_var_scope="cls"):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(model.get_embedding_output())

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    double_forward_loss = (
        kl_divergence_w_logits(start_logits, d_start_logits, reverse_kl=True) +
        kl_divergence_w_logits(end_logits, d_end_logits, reverse_kl=True)) / 2.0

    return double_forward_loss


def compute_double_forward_loss_v3(start_logits, end_logits, model, output_var_scope="cls"):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(model.get_embedding_output())

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    double_forward_loss = (
        kl_divergence_w_logits(start_logits, d_start_logits) +
        kl_divergence_w_logits(end_logits, d_end_logits)) / 2.0


def l2_normalize_vector(d, epsilon1=1e-12, epsilon2=1e-6):
  """Normalizes vector."""
  d /= (epsilon1 + tf.reduce_max(tf.abs(d), axis=-1, keep_dims=True))
  d /= tf.sqrt(epsilon2 + tf.reduce_sum(tf.square(d), axis=-1, keep_dims=True))
  return d


def l1_normalize_vector(d, epsilon=1e-6):
    """Normalizes vector in L1."""
    d /= (epsilon + tf.reduce_sum(tf.abs(d), axis=-1, keep_dims=True))
    return d


def linf_normalize_vector(d, epsilon=1e-6):
    """Normalizes vector in Linf."""
    d /= (epsilon + tf.reduce_max(tf.abs(d), axis=-1, keep_dims=True))
    return d


def normalize_vector(d, normalizer="L2"):
  if normalizer == "L2":
    return l2_normalize_vector(d)
  elif normalizer == "L1":
    return l1_normalize_vector(d)
  elif normalizer == "Linf":
    return linf_normalize_vector(d)
  else:
      raise ValueError("Unknown normalizer %s" % normalizer)


def generate_noise(x, normalizer="L2"):
  d = tf.random_normal(shape=tf.shape(x))
  return normalize_vector(d, normalizer=normalizer)


def get_embeddings(model, noise_normalizer, noise_epsilon,
                   learn_noise_network=False):
    """Gets the model embeddings with or without random noise."""
    if noise_epsilon > 0:
        if learn_noise_network:
            # TODO(chenghao): Make this learnable.
            pass
        else:
            rand_noise = tf.stop_gradient(
                noise_epsilon * generate_noise(
                    model.get_embedding_output(), normalizer=noise_normalizer))
            embeddings = model.get_embedding_output() + rand_noise
    else:
        embeddings = model.get_embedding_output()

    return embeddings


def compute_jacobian_loss(model, output_var_scope="cls"):
    """Computes Jacobian loss."""
    with tf.GradientTape(persistent=True) as tape:
        embedding_inputs = model.get_embedding_output()
        tape.watch(embedding_inputs)

        final_hidden, _ = model.adv_forward(embedding_inputs)
        (start_logits, end_logits) = compute_forward_logits(final_hidden)

    start_jacobian = tape.batch_jacobian(start_logits, embedding_inputs)
    start_jacobian_loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(start_jacobian), axis=tf.range(1, tf.rank(start_jacobian))))

    end_jacobian = tape.batch_jacobian(end_logits, embedding_inputs)
    end_jacobian_loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(end_jacobian), axis=tf.range(1, tf.rank(end_jacobian))))

    return (start_jacobian_loss + end_jacobian_loss) / 2.0


def compute_approx_jacobian_loss(start_logits, end_logits, inputs):
    """Computes approximate Jacobian loss."""
    start_jacobian_loss = compute_approx_jacobian_norm(start_logits, inputs)
    end_jacobian_loss = compute_approx_jacobian_norm(end_logits, inputs)

    return (start_jacobian_loss + end_jacobian_loss) / 2.0


def compute_approx_jacobian_norm(logits, inputs):
    """Computes approximate Frobenius norm of Jacobian."""
    random_vector = generate_noise(logits)
    logits_shape = logits.shape.as_list()
    batch_size = logits_shape[0]

    flat_logits = tf.reshape(logits, [-1])
    flat_rand = tf.reshape(random_vector, [-1])

    sampled_jacobian = tf.gradients(flat_logits * flat_rand, [inputs])[0]

    jacobian_norm = tf.reduce_sum(
        tf.square(sampled_jacobian)) / tf.to_float(batch_size)

    return jacobian_norm


def alpha_beta_kl_divergence_with_logits(
    logits_p, logits_q, alpha=1.0, beta=1.0, label_weights=None, axis=-1,
    epsilon=1e-6,
):
    """Computes alpha-beta KL divergence."""
    logits_p = tf.cast(logits_p, dtype=tf.float32)
    logits_q = tf.cast(logits_q, dtype=tf.float32)

    log_p = tf.nn.log_softmax(logits_p, axis=axis)
    log_q = tf.nn.log_softmax(logits_q, axis=axis)

    p = tf.nn.softmax(logits_p, axis=axis)

    nonbp_p = tf.stop_gradient(p)
    nonbp_log_p = tf.stop_gradient(log_p)

    kl_div = tf.reduce_sum(
        alpha * p * nonbp_log_p  - beta * nonbp_p * log_q, axis=axis)

    if label_weights is not None:
        label_weights = tf.reshape(label_weights, [-1])
        num = tf.reduce_sum(kl_div * label_weights)
        dem = tf.reduce_sum(label_weights) + epsilon
        kl_div = num / dem
    else:
        kl_div = tf.reduce_mean(kl_div)

    return kl_div


def compute_double_forward_loss_w_add_noise_topk_pos_only(
        start_logits, end_logits, model, loss_type="v3", alpha=1.0, beta=1.0,
        noise_normalizer="L2", noise_epsilon=1e-5, output_var_scope="cls", k=20):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    # Gets topk start and end positions.
    _, start_positions = tf.nn.top_k(start_logits, k=k)
    _, end_positions = tf.nn.top_k(end_logits, k=k)

    start_log_probs = tf.nn.log_softmax(start_logits)
    end_log_probs = tf.nn.log_softmax(end_logits)

    d_start_log_probs = tf.nn.log_softmax(d_start_logits)
    d_end_log_probs = tf.nn.log_softmax(d_end_logits)

    start_top_k_log_probs = batch_gather(start_log_probs, start_positions)
    end_top_k_log_probs = batch_gather(end_log_probs, end_positions)

    d_start_top_k_log_probs = batch_gather(d_start_log_probs, start_positions)
    d_end_top_k_log_probs = batch_gather(d_end_log_probs, end_positions)

    # if loss_type == "v1":
    #     tf.logging.info("Using double forward loss v1")
    #     double_forward_loss = (
    #         symmetric_kl(start_logits, d_start_logits) +
    #         symmetric_kl(end_logits, d_end_logits)) / 2.0
    # elif loss_type == "v2":
    #     tf.logging.info("Using double forward loss v2")
    #     double_forward_loss = (
    #         kl_divergence_w_logits(start_logits, d_start_logits, reverse_kl=True) +
    #         kl_divergence_w_logits(end_logits, d_end_logits, reverse_kl=True)) / 2.0
    if loss_type == "v3":
        tf.logging.info("Using double forward loss topk v3")
        double_forward_loss = 0.5 * (
            tf.reduce_mean(kl_divergence_w_log_prob(
                start_top_k_log_probs, d_start_top_k_log_probs)) +
            tf.reduce_mean(kl_divergence_w_log_prob(
                end_top_k_log_probs, d_end_top_k_log_probs))
        )
    elif loss_type == "hellinger":
        tf.logging.info("Using double forward loss topk with squared hellinger loss")
        double_forward_loss = 0.5 * (
            tf.reduce_mean(hellinger_distance_w_log_prob(
                start_top_k_log_probs, d_start_top_k_log_probs)) +
            tf.reduce_mean(hellinger_distance_w_log_prob(
                end_top_k_log_probs, d_end_top_k_log_probs))
        )
    elif loss_type == "js":
        tf.logging.info("Using double forward topk with Jensen-Shannon divergence")
        double_forward_loss = 0.5 * (
            tf.reduce_mean(js_divergence_w_log_prob(
                start_top_k_log_probs, d_start_top_k_log_probs)) +
            tf.reduce_mean(js_divergence_w_log_prob(
                end_top_k_log_probs, d_end_top_k_log_probs))
        )
    else:
        raise ValueError("Unknown loss type %s" % loss_type)

    return double_forward_loss



def compute_double_forward_loss_w_add_noise(
        start_logits, end_logits, model, loss_type="v3", alpha=1.0, beta=1.0,
        noise_normalizer="L2", noise_epsilon=1e-5, output_var_scope="cls"):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    if loss_type == "v1":
        tf.logging.info("Using double forward loss v1")
        double_forward_loss = (
            symmetric_kl(start_logits, d_start_logits) +
            symmetric_kl(end_logits, d_end_logits)) / 2.0
    elif loss_type == "v2":
        tf.logging.info("Using double forward loss v2")
        double_forward_loss = (
            kl_divergence_w_logits(start_logits, d_start_logits, reverse_kl=True) +
            kl_divergence_w_logits(end_logits, d_end_logits, reverse_kl=True)) / 2.0
    elif loss_type == "v3":
        tf.logging.info("Using double forward loss v3")
        double_forward_loss = (
            kl_divergence_w_logits(start_logits, d_start_logits) +
            kl_divergence_w_logits(end_logits, d_end_logits)) / 2.0
    elif loss_type == "alpha_beta":
        tf.logging.info("Using alpha-beta KL divergence")
        double_forward_loss = 0.5 * (
            alpha_beta_kl_divergence_with_logits(
                start_logits, d_start_logits, alpha=alpha, beta=beta) +
            alpha_beta_kl_divergence_with_logits(
                end_logits, d_end_logits, alpha=alpha, beta=beta)
        )
    else:
        raise ValueError("Unknown loss type %s" % loss_type)

    return double_forward_loss


def compute_double_forward_loss_v1_w_add_noise(
        start_logits, end_logits, model,
        noise_normalizer="L2", noise_epsilon=1e-5, output_var_scope="cls"):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    double_forward_loss = (
        symmetric_kl(start_logits, d_start_logits) +
        symmetric_kl(end_logits, d_end_logits)) / 2.0

    return double_forward_loss


def hellinger_distance(logits, d_logits, axis=-1):
    log_probs = tf.nn.log_softmax(logits)
    d_log_probs = tf.nn.log_softmax(d_logits)
    probs = tf.exp(0.5 * log_probs)
    d_probs = tf.exp(0.5 * d_log_probs)
    loss = 0.5 * tf.reduce_sum(
        tf.square(probs - d_probs), axis=axis)
    return loss


def js_divergence(logits, d_logits, axis=-1):
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    d_probs = tf.nn.softmax(d_logits)
    d_log_probs = tf.nn.log_softmax(d_logits)
    mean_logp = tf.reduce_logsumexp(
        tf.concat([tf.expand_dims(log_probs, axis=-1),
                   tf.expand_dims(d_log_probs, axis=-1)], axis=-1), axis=-1)
    mean_logp += tf.log(0.5)
    loss = 0.5 * (
        kl_divergence(probs, log_probs, mean_logp, axis=axis) +
        kl_divergence(d_probs, d_log_probs, mean_logp, axis=axis))
    return loss


def single_doc_logits(logits):
    return tf.reshape(logits, [1, -1])


def kl_with_mask(log_p, log_q, log_p_offset=0, log_q_offset=0, mask=None, axis=-1):
    p = tf.exp(log_p)
    p_offset = tf.exp(log_p_offset)
    if mask is not None:
        p *= tf.stop_gradient(mask)
    return tf.reduce_sum(p * (log_p - log_q), axis=axis
    ) + tf.reduce_sum(p_offset * (log_p_offset - log_q_offset), axis=axis)


def hellinger_with_mask(log_p, log_q, log_p_offset=0, log_q_offset=0, mask=None, axis=-1):
    p = tf.exp(0.5 * log_p)
    p_offset = tf.exp(0.5 * log_p_offset)
    q = tf.exp(0.5 * log_q)
    q_offset = tf.exp(0.5 * log_q_offset)
    if mask is not None:
        return tf.reduce_sum(mask * tf.square(p - q), axis=axis) + tf.reduce_sum(
            tf.square(p_offset - q_offset), axis=axis)
    return tf.reduce_sum(tf.square(p - q), axis=axis) + tf.reduce_sum(
        tf.square(p_offset - q_offset), axis=axis)


def js_with_mask(log_p, log_q, log_p_offset=0, log_q_offset=0, mask=None, axis=-1):
    mean_logp = tf.reduce_logsumexp(
        tf.concat([tf.expand_dims(log_p, axis=-1),
                   tf.expand_dims(log_q, axis=-1)], axis=-1), axis=-1)
    mean_logp += tf.log(0.5)
    mean_logp_offset = tf.reduce_logsumexp(
        tf.concat([tf.expand_dims(log_p_offset, axis=-1),
                   tf.expand_dims(log_q_offset, axis=-1)], axis=-1), axis=-1)
    mean_logp_offset += tf.log(0.5)
    return 0.5 * (
        kl_with_mask(log_p, mean_logp, log_p_offset=log_p_offset,
                     log_q_offset=mean_logp_offset, mask=mask, axis=axis) +
        kl_with_mask(log_q, mean_logp, log_p_offset=log_q_offset,
                     log_q_offset=mean_logp_offset, mask=mask, axis=axis))


def compute_topk_span_logprob_with_mask(start_logits, end_logits, k=10,
                                        span_ub=20, null_pos=0, max_k=100):
    """Computes the top-k span logits and corresponding mask.
    Args:
        start_logits: A float tensor of shape [batch_size, max_seq_length].
        end_logits: A float tensor of shape [batch_size, max_seq_length].

    Returns:
        topk_span_log_probs: A float tensor of shape [batch_size, k].
        valid_span_mask: A float tensor of shape [batch_size, k].
        start_positions: A integer tensor of shape[batch_size, k].
        end_positions: A integer tensor of shape[batch_size, k].
    """
    # Gets topk start and end positions.
    bs, max_seq_length = modeling.get_shape_list(start_logits, expected_rank=2)
    # null_pos_mask = tf.tile(
    #     tf.one_hot([null_pos], depth=max_seq_length, dtype=tf.float32),
    #     [bs, 1]) * LARGE_VAL
    # _, start_positions = tf.nn.top_k(start_logits + null_pos_mask, k=k)
    # _, end_positions = tf.nn.top_k(end_logits + null_pos_mask, k=k)

    _, start_positions = tf.nn.top_k(start_logits, k=k)
    _, end_positions = tf.nn.top_k(end_logits, k=k)

    # This will be of shape [batch_size, k, 1].
    exp_starts = tf.expand_dims(start_positions, axis=-1)

    # This will be of shape [batch_size, 1, k].
    exp_ends = tf.expand_dims(end_positions, axis=1)

    # Tensors of shape [batch_size, k, k].
    tiled_start_indices = tf.tile(exp_starts, [1, 1, k])
    tiled_end_indices = tf.tile(exp_ends, [1, k, 1])

    # This will be of shape [batch_size, k, k].
    start_before_end = tiled_end_indices - tiled_start_indices

    # A tensor of shape [batch_size, k, k], indicating the start-end pair is
    # valid, i.e. start before end and span length is no longer than ub.
    valid_span_mask = tf.stop_gradient(tf.cast(
        start_before_end >= 0, dtype=tf.float32) * tf.cast(
            start_before_end <= span_ub, dtype=tf.float32))

    bs, _, _ = modeling.get_shape_list(valid_span_mask, expected_rank=3)
    valid_span_mask = tf.reshape(valid_span_mask, [bs, k*k])

    # This will be of shape [batch_size, k*k].
    topk_span_logits = compute_span_score(
        start_logits,
        end_logits,
        start_positions,
        end_positions,
        k=k,
        axis=-1,
    )

    # Increases the topk by 10 factor, but up limits it by max_k.
    span_k = min(10 * k, min(max_k, k * k))

    tf.logging.info(f"Using top-{span_k} spans for VAT")
    _, topk_span_indices = tf.nn.top_k(
        topk_span_logits + tf.log(valid_span_mask), k=span_k)

    # This tensor is of shape [batch_size, k, k*k].
    topk_span_mask = tf.one_hot(topk_span_indices, depth=k*k, dtype=tf.float32)
    topk_span_mask = tf.reduce_max(topk_span_mask, axis=1)
    topk_span_mask *= valid_span_mask

    return topk_span_logits, topk_span_mask, start_positions, end_positions


def compute_span_vat_loss(start_logits, end_logits, model,
                          loss_type="v3",
                          k=10, span_ub=20,
                          output_var_scope="cls",
                          noise_normalizer="L2",
                          vat_type="global",
                          valid_span_only=False,
                          rand_noise_epsilon=1e-3,
                          global_log_prob=False,
                          noise_epsilon=1e-3):
    """Computes the double forward loss."""
    rand_noise = rand_noise_epsilon * generate_noise(
            model.get_embedding_output(), normalizer="L2")
    embeddings = model.get_embedding_output() + rand_noise
    final_hidden, _ = model.adv_forward(embeddings)

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    (clean_span_logits, valid_span_mask, start_positions, end_positions
     ) = compute_topk_span_logprob_with_mask(
        start_logits, end_logits, k=k, span_ub=span_ub)

    noise_span_logits = compute_span_score(
        d_start_logits, d_end_logits, start_positions, end_positions, k=k, axis=-1)

    loss_func = kl_with_mask
    if loss_type == "js":
        tf.logging.info("Using span VAT with Jensen-Shannon divergence")
        loss_func = js_with_mask
    elif loss_type == "hellinger":
        tf.logging.info("Using span VAT with squared hellinger loss")
        loss_func = hellinger_with_mask
    else:
        tf.logging.info("Using span VAT with KL-divergence")

    perturb_loss = 0.0
    if vat_type == "global" or vat_type == "global_local":
        clean_z_log = (tf.reduce_logsumexp(single_doc_logits(start_logits)) +
                       tf.reduce_logsumexp(single_doc_logits(end_logits)))
        noise_z_log = (tf.reduce_logsumexp(single_doc_logits(d_start_logits)) +
                       tf.reduce_logsumexp(single_doc_logits(d_end_logits)))

        clean_span_log_prob = single_doc_logits(clean_span_logits - clean_z_log)
        noise_span_log_prob = single_doc_logits(noise_span_logits - noise_z_log)

        clean_non_topk_log_prob = tf.stop_gradient(
            tf.log(1.0 - tf.reduce_sum(
                tf.exp(clean_span_log_prob) * single_doc_logits(valid_span_mask), keep_dims=True)))
        noise_non_topk_log_prob = tf.stop_gradient(
            tf.log(1.0 - tf.reduce_sum(
                tf.exp(noise_span_log_prob) * single_doc_logits(valid_span_mask), keep_dims=True)))

        perturb_loss += tf.reduce_mean(loss_func(
            clean_span_log_prob, noise_span_log_prob,
            log_p_offset=clean_non_topk_log_prob,
            log_q_offset=noise_non_topk_log_prob,
            mask=single_doc_logits(valid_span_mask)))
    elif vat_type == "local" or vat_type == "global_local":
        clean_z_log = (tf.reduce_logsumexp(start_logits, axis=-1, keep_dims=True) +
                       tf.reduce_logsumexp(end_logits, axis=-1, keep_dims=True))
        noise_z_log = (tf.reduce_logsumexp(d_start_logits, axis=-1, keep_dims=True) +
                       tf.reduce_logsumexp(d_end_logits, axis=-1, keep_dims=True))

        clean_span_log_prob = clean_span_logits - clean_z_log
        noise_span_log_prob = noise_span_logits - noise_z_log

        clean_non_topk_log_prob = tf.stop_gradient(
            tf.log(1.0 - tf.reduce_sum(tf.exp(clean_span_log_prob) * valid_span_mask,
                                       keep_dims=True, axis=-1)))
        noise_non_topk_log_prob = tf.stop_gradient(
            tf.log(1.0 - tf.reduce_sum(tf.exp(noise_span_log_prob) * valid_span_mask,
                                       keep_dims=True, axis=-1)))

        perturb_loss += tf.reduce_mean(loss_func(
            clean_span_log_prob,
            noise_span_log_prob,
            log_p_offset=clean_non_topk_log_prob,
            log_q_offset=noise_non_topk_log_prob,
            mask=valid_span_mask))
    else:
        raise ValueError("Unknown VAT type %s" % vat_type)

    perturb = tf.stop_gradient(tf.gradients(perturb_loss, [rand_noise])[0])
    perturb = noise_epsilon * normalize_vector(perturb, normalizer=noise_normalizer)

    embeddings = model.get_embedding_output() + perturb
    final_hidden, _ = model.adv_forward(embeddings)
    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)
    noise_span_logits = compute_span_score(
        d_start_logits, d_end_logits, start_positions, end_positions, k=k)

    vat_loss = 0.0
    if vat_type == "global" or vat_type == "global_local":
        tf.logging.info("Using global span VAT")
        clean_z_log = (tf.reduce_logsumexp(single_doc_logits(start_logits)) +
                       tf.reduce_logsumexp(single_doc_logits(end_logits)))
        noise_z_log = (tf.reduce_logsumexp(single_doc_logits(d_start_logits)) +
                       tf.reduce_logsumexp(single_doc_logits(d_end_logits)))

        clean_span_log_prob = single_doc_logits(clean_span_logits - clean_z_log)
        noise_span_log_prob = single_doc_logits(noise_span_logits - noise_z_log)

        clean_non_topk_log_prob = tf.stop_gradient(
            tf.log(1.0 - tf.reduce_sum(
                tf.exp(clean_span_log_prob) * single_doc_logits(valid_span_mask),
                keep_dims=True)))
        noise_non_topk_log_prob = tf.stop_gradient(
            tf.log(1.0 - tf.reduce_sum(
                tf.exp(noise_span_log_prob) * single_doc_logits(valid_span_mask),
                keep_dims=True)))

        vat_loss += tf.reduce_mean(loss_func(
            clean_span_log_prob, noise_span_log_prob,
            log_p_offset=clean_non_topk_log_prob,
            log_q_offset=noise_non_topk_log_prob,
            mask=single_doc_logits(valid_span_mask)))
    elif vat_type == "local" or vat_type == "global_local":
        tf.logging.info("Using local span VAT")
        clean_z_log = (tf.reduce_logsumexp(start_logits, axis=-1, keep_dims=True) +
                       tf.reduce_logsumexp(end_logits, axis=-1, keep_dims=True))
        noise_z_log = (tf.reduce_logsumexp(d_start_logits, axis=-1, keep_dims=True) +
                       tf.reduce_logsumexp(d_end_logits, axis=-1, keep_dims=True))

        clean_span_log_prob = clean_span_logits - clean_z_log
        noise_span_log_prob = noise_span_logits - noise_z_log

        clean_non_topk_log_prob = tf.stop_gradient(
            tf.log(1.0 - tf.reduce_sum(tf.exp(clean_span_log_prob) * valid_span_mask,
                                       keep_dims=True, axis=-1)))
        noise_non_topk_log_prob = tf.stop_gradient(
            tf.log(1.0 - tf.reduce_sum(tf.exp(noise_span_log_prob) * valid_span_mask,
                                       keep_dims=True, axis=-1)))

        vat_loss += tf.reduce_mean(loss_func(
            clean_span_log_prob,
            noise_span_log_prob,
            log_p_offset=clean_non_topk_log_prob,
            log_q_offset=noise_non_topk_log_prob,
            mask=valid_span_mask))
    else:
        raise ValueError("Unknown VAT type %s" % vat_type)

    return vat_loss


def perturb_estimation(start_logits, end_logits, model, loss_func, rand_noise,
                       vat_type="global", k=1, noise_epsilon=1e-3,
                       rand_noise_epsilon=1e-3, accum_est=False,
                       noise_normalizer="L2"):
    """Iterative estimation of perturbation."""
    perturb = rand_noise_epsilon * rand_noise if accum_est else rand_noise
    for _ in range(k):
        if accum_est:
            embeddings = model.get_embedding_output() + perturb
        else:
            embeddings = model.get_embedding_output() + rand_noise_epsilon * perturb

        final_hidden, _ = model.adv_forward(embeddings)

        (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

        if vat_type == "global":
            perturb_loss = tf.reduce_sum(
                loss_func(single_doc_logits(start_logits),
                          single_doc_logits(d_start_logits)) +
                loss_func(single_doc_logits(end_logits),
                          single_doc_logits(d_end_logits))) / 2.0
        elif vat_type == "local":
            perturb_loss = tf.reduce_mean(
                loss_func(start_logits, d_start_logits) +
                loss_func(end_logits, d_end_logits)) / 2.0
        elif vat_type == "global_local":
            perturb_loss = tf.reduce_mean(
                loss_func(start_logits, d_start_logits) +
                loss_func(end_logits, d_end_logits)) / 2.0
            perturb_loss += tf.reduce_sum(
                loss_func(single_doc_logits(start_logits),
                          single_doc_logits(d_start_logits)) +
                loss_func(single_doc_logits(end_logits),
                          single_doc_logits(d_end_logits))) / 2.0
        else:
            raise ValueError("Unknown VAT type %s" % vat_type)

        new_perturb = tf.stop_gradient(tf.gradients(perturb_loss, [perturb])[0])
        new_perturb = normalize_vector(new_perturb, normalizer=noise_normalizer)

        if accum_est:
            perturb += noise_epsilon * new_perturb
        else:
            perturb = new_perturb

    return perturb


def topk_perturb_estimation(start_logits, end_logits, model, loss_func,
                            rand_noise, topk_start_positions,
                            topk_end_positions, vat_type="global", k=1,
                            noise_epsilon=1e-3,
                            rand_noise_epsilon=1e-3, accum_est=False,
                            noise_normalizer="L2"):
    """Iterative estimation of perturbation."""
    perturb = rand_noise_epsilon * rand_noise if accum_est else rand_noise
    topk_start_logits = batch_gather(start_logits, topk_start_positions)
    topk_end_logits = batch_gather(end_logits, topk_end_positions)

    for _ in range(k):
        if accum_est:
            embeddings = model.get_embedding_output() + perturb
        else:
            embeddings = model.get_embedding_output() + rand_noise_epsilon * perturb

        final_hidden, _ = model.adv_forward(embeddings)

        (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

        d_topk_start_logits = batch_gather(d_start_logits, topk_start_positions)
        d_topk_end_logits = batch_gather(d_end_logits, topk_end_positions)

        perturb_loss = 0.0
        if vat_type == "global" or vat_type == "global_local":
            start_z = tf.reduce_logsumexp(start_logits, keepdims=True)
            end_z = tf.reduce_logsumexp(end_logits, keepdims=True)
            topk_start_log_prob = topk_start_logits - start_z
            topk_end_log_prob = topk_end_logits - end_z
            nontopk_start_log_prob = tf.stop_gradient(tf.log(
                1.0 - tf.reduce_sum(tf.exp(topk_start_log_prob), keepdims=True)
            ))
            nontopk_end_log_prob = tf.stop_gradient(tf.log(
                1.0 - tf.reduce_sum(tf.exp(topk_end_log_prob), keepdims=True)
            ))

            d_start_z = tf.reduce_logsumexp(d_start_logits, keepdims=True)
            d_end_z = tf.reduce_logsumexp(d_end_logits, keepdims=True)
            d_topk_start_log_prob = d_topk_start_logits - d_start_z
            d_topk_end_log_prob = d_topk_end_logits - d_end_z
            d_nontopk_start_log_prob = tf.stop_gradient(tf.log(
                1.0 - tf.reduce_sum(tf.exp(d_topk_start_log_prob), keepdims=True)
            ))
            d_nontopk_end_log_prob = tf.stop_gradient(tf.log(
                1.0 - tf.reduce_sum(tf.exp(d_topk_end_log_prob), keepdims=True)
            ))

            perturb_loss += tf.reduce_mean(
                loss_func(single_doc_logits(topk_start_log_prob),
                          single_doc_logits(d_topk_start_log_prob),
                          log_p_offset=nontopk_start_log_prob,
                          log_q_offset=d_nontopk_start_log_prob) +
                loss_func(single_doc_logits(topk_end_log_prob),
                          single_doc_logits(d_topk_end_log_prob),
                          log_p_offset=nontopk_end_log_prob,
                          log_q_offset=d_nontopk_end_log_prob)) / 2.0
        elif vat_type == "local" or vat_type == "global_local":
            start_z = tf.reduce_logsumexp(start_logits, axis=-1, keepdims=True)
            end_z = tf.reduce_logsumexp(end_logits, axis=-1, keepdims=True)
            topk_start_log_prob = topk_start_logits - start_z
            topk_end_log_prob = topk_end_logits - end_z
            nontopk_start_log_prob = tf.stop_gradient(tf.log(
                1.0 - tf.reduce_sum(
                    tf.exp(topk_start_log_prob), axis=-1, keepdims=True)))
            nontopk_end_log_prob = tf.stop_gradient(tf.log(
                1.0 - tf.reduce_sum(
                    tf.exp(topk_end_log_prob), axis=-1, keepdims=True)))


            d_start_z = tf.reduce_logsumexp(d_start_logits, axis=-1, keepdims=True)
            d_end_z = tf.reduce_logsumexp(d_end_logits, axis=-1, keepdims=True)
            d_topk_start_log_prob = d_topk_start_logits - d_start_z
            d_topk_end_log_prob = d_topk_end_logits - d_end_z
            d_nontopk_start_log_prob = tf.stop_gradient(tf.log(
                1.0 - tf.reduce_sum(
                    tf.exp(d_topk_start_log_prob), axis=-1, keepdims=True)
            ))
            d_nontopk_end_log_prob = tf.stop_gradient(tf.log(
                1.0 - tf.reduce_sum(
                    tf.exp(d_topk_end_log_prob), axis=-1, keepdims=True)
            ))

            perturb_loss += tf.reduce_mean(
                loss_func(start_logits, d_start_logits,
                          log_p_offset=nontopk_start_log_prob,
                          log_q_offset=d_nontopk_start_log_prob) +
                loss_func(end_logits, d_end_logits,
                          log_p_offset=nontopk_end_log_prob,
                          log_q_offset=d_nontopk_end_log_prob)) / 2.0
        else:
            raise ValueError("Unknown VAT type %s" % vat_type)

        new_perturb = tf.stop_gradient(tf.gradients(perturb_loss, [perturb])[0])
        new_perturb = normalize_vector(new_perturb, normalizer=noise_normalizer)

        if accum_est:
            perturb += noise_epsilon * new_perturb
        else:
            perturb = new_perturb

    return perturb


def compute_topk_vat_loss(start_logits, end_logits, model,
                          loss_type="v3",
                          output_var_scope="cls",
                          noise_normalizer="L2",
                          vat_type="global",
                          rand_noise_epsilon=1e-3,
                          noise_epsilon=1e-3,
                          top_k=20,
                          use_topk_for_reg=True,
                          null_pos=0,
                          accum_est=False, num_est_iter=1):
    """Computes the topk position based VAT loss."""
    rand_noise = generate_noise(
            model.get_embedding_output(), normalizer=noise_normalizer)

    loss_func = kl_with_mask
    vat_loss_func = kl_divergence_w_logits
    if loss_type == "js":
        tf.logging.info("Using topk VAT with Jensen-Shannon divergence")
        loss_func = js_with_mask
        vat_loss_func = js_divergence
    elif loss_type == "hellinger":
        tf.logging.info("Using topk VAT with squared hellinger loss")
        loss_func = hellinger_with_mask
        vat_loss_func = hellinger_distance
    else:
        tf.logging.info("Using topk VAT with KL-divergence")

    # Gets topk start and end positions.
    # bs, max_seq_length = modeling.get_shape_list(start_logits, expected_rank=2)
    # null_pos_mask = tf.tile(
    #     tf.one_hot([null_pos], depth=max_seq_length, dtype=tf.float32),
    #     [bs, 1]) * LARGE_VAL
    # _, start_positions = tf.nn.top_k(start_logits + null_pos_mask, k=top_k)
    # _, end_positions = tf.nn.top_k(end_logits + null_pos_mask, k=top_k)
    _, topk_start_positions = tf.nn.top_k(start_logits, k=top_k)
    _, topk_end_positions = tf.nn.top_k(end_logits, k=top_k)


    # start_top_k_logits = batch_gather(start_logits, start_positions)
    # end_top_k_logits = batch_gather(end_logits, end_positions)

    perturb = topk_perturb_estimation(
        start_logits,
        end_logits,
        model,
        loss_func,
        rand_noise,
        topk_start_positions,
        topk_end_positions,
        vat_type=vat_type,
        k=num_est_iter,
        rand_noise_epsilon=rand_noise_epsilon,
        noise_epsilon=noise_epsilon,
        accum_est=accum_est,
        noise_normalizer=noise_normalizer,
    )

    r_adv = perturb if accum_est else noise_epsilon * perturb
    r_adv = tf.stop_gradient(r_adv)

    embeddings = model.get_embedding_output() + r_adv
    final_hidden, _ = model.adv_forward(embeddings)
    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    # if use_topk_for_reg:
    #     start_logits = start_top_k_logits
    #     end_logits = end_top_k_logits
    #     d_start_logits = batch_gather(d_start_logits, start_positions)
    #     d_end_logits = batch_gather(d_end_logits, end_positions)

    topk_start_logits = batch_gather(start_logits, topk_start_positions)
    topk_end_logits = batch_gather(end_logits, topk_end_positions)
    d_topk_start_logits = batch_gather(d_start_logits, topk_start_positions)
    d_topk_end_logits = batch_gather(d_end_logits, topk_end_positions)

    vat_loss = 0.0
    # if vat_type == "global":
    #     tf.logging.info("Using global topk VAT loss")
    #     vat_loss = tf.reduce_mean(
    #         vat_loss_func(single_doc_logits(start_logits),
    #                       single_doc_logits(d_start_logits)) +
    #         vat_loss_func(single_doc_logits(end_logits),
    #                       single_doc_logits(d_end_logits))) / 2.0
    # elif vat_type == "local":
    #     tf.logging.info("Using local VAT loss")
    #     vat_loss = tf.reduce_mean(
    #         vat_loss_func(start_logits, d_start_logits) +
    #         vat_loss_func(end_logits, d_end_logits)) / 2.0
    # elif vat_type == "global_local":
    #     tf.logging.info("Using global_local VAT loss")
    #     vat_loss = tf.reduce_mean(
    #         vat_loss_func(start_logits, d_start_logits) +
    #         vat_loss_func(end_logits, d_end_logits)) / 2.0
    #     vat_loss += tf.reduce_mean(
    #         vat_loss_func(single_doc_logits(start_logits),
    #                       single_doc_logits(d_start_logits)) +
    #         vat_loss_func(single_doc_logits(end_logits),
    #                       single_doc_logits(d_end_logits))) / 2.0
    # else:
    #     raise ValueError("Unknown vat_type %s" % vat_type)

    if vat_type == "global" or vat_type == "global_local":
        tf.logging.info("Using global topk VAT loss")
        start_z = tf.reduce_logsumexp(start_logits, keepdims=True)
        end_z = tf.reduce_logsumexp(end_logits, keepdims=True)
        topk_start_log_prob = topk_start_logits - start_z
        topk_end_log_prob = topk_end_logits - end_z
        nontopk_start_log_prob = tf.stop_gradient(tf.log(
            1.0 - tf.reduce_sum(tf.exp(topk_start_log_prob), keepdims=True)
        ))
        nontopk_end_log_prob = tf.stop_gradient(tf.log(
            1.0 - tf.reduce_sum(tf.exp(topk_end_log_prob), keepdims=True)
        ))

        d_start_z = tf.reduce_logsumexp(d_start_logits, keepdims=True)
        d_end_z = tf.reduce_logsumexp(d_end_logits, keepdims=True)
        d_topk_start_log_prob = d_topk_start_logits - d_start_z
        d_topk_end_log_prob = d_topk_end_logits - d_end_z
        d_nontopk_start_log_prob = tf.stop_gradient(tf.log(
            1.0 - tf.reduce_sum(tf.exp(d_topk_start_log_prob), keepdims=True)
        ))
        d_nontopk_end_log_prob = tf.stop_gradient(tf.log(
            1.0 - tf.reduce_sum(tf.exp(d_topk_end_log_prob), keepdims=True)
        ))

        vat_loss += tf.reduce_mean(
            loss_func(single_doc_logits(topk_start_log_prob),
                      single_doc_logits(d_topk_start_log_prob),
                      log_p_offset=nontopk_start_log_prob,
                      log_q_offset=d_nontopk_start_log_prob) +
            loss_func(single_doc_logits(topk_end_log_prob),
                      single_doc_logits(d_topk_end_log_prob),
                      log_p_offset=nontopk_end_log_prob,
                      log_q_offset=d_nontopk_end_log_prob)) / 2.0
    elif vat_type == "local" or vat_type == "global_local":
        tf.logging.info("Using local VAT loss")
        start_z = tf.reduce_logsumexp(start_logits, axis=-1, keepdims=True)
        end_z = tf.reduce_logsumexp(end_logits, axis=-1, keepdims=True)
        topk_start_log_prob = topk_start_logits - start_z
        topk_end_log_prob = topk_end_logits - end_z
        nontopk_start_log_prob = tf.stop_gradient(tf.log(
            1.0 - tf.reduce_sum(
                tf.exp(topk_start_log_prob), axis=-1, keepdims=True)))
        nontopk_end_log_prob = tf.stop_gradient(tf.log(
            1.0 - tf.reduce_sum(
                tf.exp(topk_end_log_prob), axis=-1, keepdims=True)))

        d_start_z = tf.reduce_logsumexp(d_start_logits, axis=-1, keepdims=True)
        d_end_z = tf.reduce_logsumexp(d_end_logits, axis=-1, keepdims=True)
        d_topk_start_log_prob = d_topk_start_logits - d_start_z
        d_topk_end_log_prob = d_topk_end_logits - d_end_z
        d_nontopk_start_log_prob = tf.stop_gradient(tf.log(
            1.0 - tf.reduce_sum(
                tf.exp(d_topk_start_log_prob), axis=-1, keepdims=True)
        ))
        d_nontopk_end_log_prob = tf.stop_gradient(tf.log(
            1.0 - tf.reduce_sum(
                tf.exp(d_topk_end_log_prob), axis=-1, keepdims=True)
        ))

        vat_loss += tf.reduce_mean(
            loss_func(start_logits, d_start_logits,
                      log_p_offset=nontopk_start_log_prob,
                      log_q_offset=d_nontopk_start_log_prob) +
            loss_func(end_logits, d_end_logits,
                      log_p_offset=nontopk_end_log_prob,
                      log_q_offset=d_nontopk_end_log_prob)) / 2.0
    else:
        raise ValueError("Unknown VAT type %s" % vat_type)

    return vat_loss


def compute_vat_loss(start_logits, end_logits, model,
                     loss_type="v3",
                     output_var_scope="cls",
                     noise_normalizer="L2",
                     vat_type="global",
                     rand_noise_epsilon=1e-3,
                     noise_epsilon=1e-3,
                     accum_est=False, num_est_iter=1):
    """Computes the double forward loss."""
    rand_noise = generate_noise(
            model.get_embedding_output(), normalizer=noise_normalizer)

    loss_func = kl_divergence_w_logits
    if loss_type == "js":
        tf.logging.info("Using VAT with Jensen-Shannon divergence")
        loss_func = js_divergence
    elif loss_type == "hellinger":
        tf.logging.info("Using VAT with squared hellinger loss")
        loss_func = hellinger_distance
    else:
        tf.logging.info("Using VAT with KL-divergence")

    perturb = perturb_estimation(
        start_logits,
        end_logits,
        model,
        loss_func,
        rand_noise,
        vat_type=vat_type,
        k=num_est_iter,
        rand_noise_epsilon=rand_noise_epsilon,
        noise_epsilon=noise_epsilon,
        accum_est=accum_est,
        noise_normalizer=noise_normalizer,
    )

    r_adv = perturb if accum_est else noise_epsilon * perturb
    r_adv = tf.stop_gradient(r_adv)

    embeddings = model.get_embedding_output() + r_adv
    final_hidden, _ = model.adv_forward(embeddings)
    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    if vat_type == "global":
        tf.logging.info("Using global VAT loss")
        vat_loss = tf.reduce_mean(
            loss_func(single_doc_logits(start_logits),
                      single_doc_logits(d_start_logits)) +
            loss_func(single_doc_logits(end_logits),
                      single_doc_logits(d_end_logits))) / 2.0
    elif vat_type == "local":
        tf.logging.info("Using local VAT loss")
        vat_loss = tf.reduce_mean(
            loss_func(start_logits, d_start_logits) +
            loss_func(end_logits, d_end_logits)) / 2.0
    elif vat_type == "global_local":
        tf.logging.info("Using global_local VAT loss")
        vat_loss = tf.reduce_mean(
            loss_func(start_logits, d_start_logits) +
            loss_func(end_logits, d_end_logits)) / 2.0
        vat_loss += tf.reduce_mean(
            loss_func(single_doc_logits(start_logits),
                      single_doc_logits(d_start_logits)) +
            loss_func(single_doc_logits(end_logits),
                      single_doc_logits(d_end_logits))) / 2.0

    return vat_loss


def compute_double_forward_loss_v2_w_add_noise(start_logits, end_logits, model,
                                               output_var_scope="cls",
                                               noise_normalizer="L2",
                                               noise_epsilon=1e-5):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    double_forward_loss = (
        kl_divergence_w_logits(start_logits, d_start_logits, reverse_kl=True) +
        kl_divergence_w_logits(end_logits, d_end_logits, reverse_kl=True)) / 2.0

    return double_forward_loss


def compute_double_forward_loss_v3_w_add_noise(start_logits, end_logits, model,
                                               output_var_scope="cls",
                                               noise_normalizer="L2",
                                               noise_epsilon=1e-5):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    double_forward_loss = (
        kl_divergence_w_logits(start_logits, d_start_logits) +
        kl_divergence_w_logits(end_logits, d_end_logits)) / 2.0

    return double_forward_loss


def is_correct_mask(logits, one_hot_labels, axis=-1):
    """Returns a 0/1 mask, 1 indicates the sample is classified correctly."""
    label_logits = tf.reduce_sum(one_hot_labels * logits, axis=axis)
    best_wrong_logits = tf.reduce_max(
        tf.log(1.0 - one_hot_labels) + logits, axis=axis
    )

    return tf.cast(label_logits > best_wrong_logits, dtype=tf.float32)


def compute_double_forward_loss_w_add_noise_correct_only(
        start_logits, end_logits, model, start_positions, end_positions,
        loss_type="v3", alpha=1.0, beta=1.0,
        noise_normalizer="L2", noise_epsilon=1e-5, output_var_scope="cls"):
    """Computes the double forward loss."""
    final_hidden, _ = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)


    start_mask = is_correct_mask(start_logits, start_positions)
    end_mask = is_correct_mask(end_logits, end_positions)

    if loss_type == "v1":
        tf.logging.info("Using double forward loss v1")
        double_forward_loss = (
            symmetric_kl(start_logits, d_start_logits, label_weights=start_mask) +
            symmetric_kl(end_logits, d_end_logits, label_weights=end_mask)) / 2.0
    elif loss_type == "v2":
        tf.logging.info("Using double forward loss v2")
        double_forward_loss = (
            kl_divergence_w_logits(
                start_logits, d_start_logits, reverse_kl=True,
                label_weights=start_mask) +
            kl_divergence_w_logits(
                end_logits, d_end_logits, reverse_kl=True,
                label_weights=end_mask)) / 2.0
    elif loss_type == "v3":
        tf.logging.info("Using double forward loss v3")
        double_forward_loss = (
            kl_divergence_w_logits(
                start_logits, d_start_logits, label_weights=start_mask) +
            kl_divergence_w_logits(
                end_logits, d_end_logits, label_weights=end_mask)) / 2.0
    elif loss_type == "alpha_beta":
        tf.logging.info("Using alpha-beta KL divergence")
        double_forward_loss = 0.5 * (
            alpha_beta_kl_divergence_with_logits(
                start_logits, d_start_logits, alpha=alpha, beta=beta,
                label_weights=start_mask) +
            alpha_beta_kl_divergence_with_logits(
                end_logits, d_end_logits, alpha=alpha, beta=beta,
                label_weights=end_mask)
        )
    else:
        raise ValueError("Unknown loss type %s" % loss_type)

    return double_forward_loss


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

def compute_target_log_prob(logits, one_hot_labels, axis=-1):
    """Computes the log-prob for the label class."""
    target_log_prob = tf.reduce_sum(
        one_hot_labels * tf.nn.log_softmax(logits, axis=-1), axis=axis,
        keep_dims=True)

    return target_log_prob


def compute_target_prob(logits, one_hot_labels, axis=-1):
    """Computes the prob for the label class."""
    target_prob = tf.reduce_sum(
        one_hot_labels * tf.nn.softmax(logits, axis=-1), axis=axis, keep_dims=True)

    return target_prob


def compute_span_score(start_scores, end_scores, start_positions, end_positions,
                       k=10, axis=-1):
    """Computes the top-k span scores."""
    start_top_k_scores = batch_gather(start_scores, start_positions)
    end_top_k_scores = batch_gather(end_scores, end_positions)

    # Tiles the topk start scores [batch_size, k, k].
    expand_start_scores = tf.tile(
        tf.expand_dims(start_top_k_scores, axis=-1), [1, 1, k])

    # Tiles the topk end probs into shape [batch_size, 1, k].
    expand_end_scores = tf.expand_dims(end_top_k_scores, axis=1)

    topk_span_scores = expand_start_scores + expand_end_scores

    bs, _, _ = modeling.get_shape_list(topk_span_scores, expected_rank=3)
    topk_span_scores = tf.reshape(topk_span_scores, [bs, k*k])
    return topk_span_scores


def compute_span_log_prob(start_logits, end_logits, start_positions,
                          end_positions, k=10, axis=-1, normalized=False):
    """Computes the top-k span log prob."""
    if normalized:
        start_log_probs = tf.nn.log_softmax(start_logits, axis=axis)
        end_log_probs = tf.nn.log_softmax(end_logits, axis=axis)
    else:
        start_log_probs = start_logits
        end_log_probs = end_logits

    topk_span_log_probs = compute_span_score(
        start_log_probs,
        end_log_probs,
        start_positions,
        end_positions,
        k=k,
        axis=axis,
    )

    return topk_span_log_probs


def compute_double_forward_loss_w_add_noise_span(
        start_logits, end_logits, model,
        loss_type="v3", alpha=1.0, beta=1.0, k=10,
        noise_normalizer="L2", noise_epsilon=1e-5, output_var_scope="cls",
        flat_logits=False):
    """Computes the double forward loss."""
    tf.logging.info("Using span-based double forward loss")
    final_hidden, _ = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    (d_start_logits, d_end_logits) = compute_forward_logits(final_hidden)

    if flat_logits:
        start_logits = single_doc_logits(start_logits)
        end_logits = single_doc_logits(end_logits)
        d_start_logits = single_doc_logits(d_start_logits)
        d_end_logits = single_doc_logits(d_end_logits)

    # Gets topk start and end positions.
    _, start_positions = tf.nn.top_k(start_logits, k=k)
    _, end_positions = tf.nn.top_k(end_logits, k=k)

    clean_span_log_prob = compute_span_log_prob(
        start_logits, end_logits, start_positions, end_positions, k=k)

    noise_span_log_prob = compute_span_log_prob(
        d_start_logits, d_end_logits, start_positions, end_positions, k=k)

    clean_span_prob = tf.exp(clean_span_log_prob)
    noise_span_prob = tf.exp(noise_span_log_prob)

    if loss_type == "v1":
        tf.logging.info("Using double forward loss v1")

        double_forward_loss = 0.5 * tf.reduce_mean(
            kl_divergence(clean_span_prob, clean_span_log_prob,
                          noise_span_log_prob) +
            rev_kl_divergence(noise_span_prob, noise_span_log_prob,
                              clean_span_log_prob)
        )
    elif loss_type == "v2":
        tf.logging.info("Using double forward loss v2")
        double_forward_loss = tf.reduce_mean(rev_kl_divergence(
            noise_span_prob, noise_span_log_prob, clean_span_log_prob))
    elif loss_type == "v3":
        tf.logging.info("Using double forward loss v3")
        double_forward_loss = tf.reduce_mean(kl_divergence(
            clean_span_prob, clean_span_log_prob, noise_span_log_prob))
    elif loss_type == "alpha_beta":
        tf.logging.info("Using alpha-beta KL divergence")
        double_forward_loss = tf.reduce_mean(
            tf.reduce_sum(
                alpha * clean_span_prob * tf.stop_gradient(clean_span_log_prob) -
                beta * tf.stop_gradient(noise_span_log_prob) * noise_span_log_prob,
                axis=-1,
            )
        )
    elif loss_type == "hellinger":
        tf.logging.info("Using double forward loss with squared hellinger loss")
        # clean_span_prob = tf.stop_gradient(clean_span_prob)
        double_forward_loss = tf.reduce_mean(
            0.5 * tf.reduce_sum(tf.square(tf.sqrt(clean_span_prob) -
                                          tf.sqrt(noise_span_prob)), axis=-1)
        )
    elif loss_type == "js":
        tf.logging.info("Using double forward with Jensen-Shannon divergence")
        # clean_span_prob = tf.stop_gradient(clean_span_prob)
        mean_prob = 0.5 * (clean_span_prob + noise_span_prob)
        mean_log_prob = tf.log(mean_prob)
        double_forward_loss = tf.reduce_mean(
            0.5 * kl_divergence(clean_span_prob, clean_span_log_prob,
                                mean_log_prob) +
            0.5 * kl_divergence(noise_span_prob, noise_span_log_prob,
                                mean_log_prob)
        )
    else:
        raise ValueError("Unknown loss type %s" % loss_type)

    return double_forward_loss


import numpy as np
import tensorflow as tf


# based on the implementation from https://github.com/passalis/probabilistic_kt
def pkt_loss(target, inputs):
    loss = cosine_similarity_loss(inputs, target)
    return loss


def cosine_similarity_loss(inputs, target, eps=1e-8):
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
    target = tf.reshape(target, [tf.shape(target)[0], -1])

    inputs = tf.nn.l2_normalize(inputs, 1)
    target = tf.nn.l2_normalize(target, 1)

    inputs_sim = (tf.matmul(inputs, inputs, transpose_b=True) + 1.0) / 2.0
    target_sim = (tf.matmul(target, target, transpose_b=True) + 1.0) / 2.0

    inputs_sim = inputs_sim / tf.reduce_sum(inputs_sim, 1)
    target_sim = target_sim / tf.reduce_sum(target_sim, 1)

    loss = tf.reduce_mean(target_sim * tf.log((target_sim + eps) / (inputs_sim + eps)))
    return loss



    




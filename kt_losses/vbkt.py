import numpy as np
import tensorflow as tf



def vbkt_loss(target, inputs):
    dim = inputs.get_shape().as_list()[1] // 2

    inputs_mu = inputs[:, :dim, :, :]
    inputs_logsigma = inputs[:, dim:, :, :]
    target_mu = target[:, :dim, :, :]
    target_logsigma = target[:, dim:, :, :]

    inputs_mu = tf.reshape(inputs_mu, [tf.shape(inputs_mu)[0], -1])
    inputs_logsigma = tf.reshape(inputs_logsigma, [tf.shape(inputs_logsigma)[0], -1])
    target_mu = tf.reshape(target_mu, [tf.shape(target_mu)[0], -1])
    target_logsigma = tf.reshape(target_logsigma, [tf.shape(target_logsigma)[0], -1])

    loss = tf.reduce_mean(0.5 * tf.math.square(inputs_mu - target_mu) / (tf.math.exp(2.0 * target_logsigma) + 1e-8))
    return loss





import numpy as np
import tensorflow as tf


def poly_kernel(x, y):
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 2)
    res = tf.pow(tf.reduce_sum(x * y, axis=-1), 2)
    return res

def nst_loss(target, inputs):
    target = tf.reshape(target, [tf.shape(target)[0], tf.shape(target)[1], -1])
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], -1])

    target = tf.nn.l2_normalize(target, 2)
    inputs = tf.nn.l2_normalize(inputs, 2)

    loss = tf.reduce_mean(poly_kernel(target, target)) \
            + tf.reduce_mean(poly_kernel(inputs, inputs)) \
            - 2 * tf.reduce_mean(poly_kernel(target, inputs)) 
    return loss


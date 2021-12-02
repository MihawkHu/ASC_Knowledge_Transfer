import numpy as np
import tensorflow as tf
import keras.backend as K

def smooth_l1_loss(x, y):
    loss = tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x-y), 1.), tf.square(x-y)/2, tf.abs(x-y)-1/2))
    return loss


def pdist(x):
    x_square = tf.reduce_sum(tf.square(x), -1)
    prod = tf.matmul(x, x, transpose_b=True)
    distance = tf.sqrt(tf.maximum(tf.expand_dims(x_square,1) + tf.expand_dims(x_square, 0) - 2 * prod, 1e-12))
    mu = tf.reduce_sum(distance)/tf.reduce_sum(tf.where(distance > 0., tf.ones_like(distance), tf.zeros_like(distance)))
    return distance / (mu + 1e-12)


def biloss(inputs, target):
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
    target = tf.reshape(target, [tf.shape(target)[0], -1])

    inputs = tf.nn.l2_normalize(inputs, 1)
    target = tf.nn.l2_normalize(target, 1)
    loss = smooth_l1_loss(pdist(inputs), pdist(target))
    return loss

def pangle(x):
    e = tf.expand_dims(x, 0) - tf.expand_dims(x, 1)
    e_norm = tf.nn.l2_normalize(e, 2)
    return tf.matmul(e_norm, e_norm, transpose_b=True)


def triloss(inputs, target):
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])
    target = tf.reshape(target, [tf.shape(target)[0], -1])

    inputs = tf.nn.l2_normalize(inputs, 1)
    target = tf.nn.l2_normalize(target, 1)
    loss = smooth_l1_loss(pangle(inputs), pangle(target))
    return loss


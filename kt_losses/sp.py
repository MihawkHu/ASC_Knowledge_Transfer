import numpy as np
import tensorflow as tf

def sp_loss(target, inputs):
    target = tf.reshape(target, [tf.shape(target)[0], -1])
    inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1])

    target_prod = tf.matmul(target, target, transpose_b=True)
    inputs_prod = tf.matmul(inputs, inputs, transpose_b=True)
    
    target_prod = tf.nn.l2_normalize(target_prod, 1)
    inputs_prod = tf.nn.l2_normalize(inputs_prod, 1)

    loss = tf.reduce_mean(tf.square(target_prod - inputs_prod))
    return loss

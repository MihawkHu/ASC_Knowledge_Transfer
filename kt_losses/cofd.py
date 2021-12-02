import numpy as np
import tensorflow as tf


def cofd_loss(target, inputs):
    neg_target = tf.math.minimum(target, 0.0)
    margin = tf.reduce_sum(neg_target, axis=[0,1,2], keepdims=True) / tf.math.count_nonzero(neg_target, axis=[0,1,2], keepdims=True, dtype=tf.float32)
    
    target = tf.math.maximum(target, margin)
    
    loss = tf.reduce_mean(tf.square(inputs - target) * tf.cast((inputs > target) | (target > 0), tf.float32))
    return loss

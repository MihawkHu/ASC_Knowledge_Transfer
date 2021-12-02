import numpy as np
import tensorflow as tf


# the implementation is based on https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/CC.py, 
def cckd_loss(target, inputs):
    delta = tf.abs(target - inputs)
    corr = tf.multiply(delta[:-1], delta[1:])
    loss = tf.reduce_mean(tf.reduce_sum(corr, axis=-1))
    return loss
    

import numpy as np
import tensorflow as tf

def at_loss(target, inputs):
    att_map_target = tf.reduce_mean(target, axis=-1)
    att_map_inputs = tf.reduce_mean(inputs, axis=-1)
    
    att_map_target = tf.nn.l2_normalize(att_map_target, [1,2])
    att_map_inputs = tf.nn.l2_normalize(att_map_inputs, [1,2])

    loss = tf.reduce_mean(tf.square(att_map_target - att_map_inputs)) / 2
    return loss

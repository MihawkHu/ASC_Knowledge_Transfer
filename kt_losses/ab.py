import numpy as np
import tensorflow as tf

def ab_loss(target, inputs):
    margin = 1.0
    
    loss_item1 = tf.math.sign(target) * tf.nn.relu(margin - inputs)
    loss_item2 = tf.math.sign(1.0 - target) * tf.nn.relu(margin + inputs)

    loss = tf.reduce_mean(loss_item1 + loss_item2)
    return loss


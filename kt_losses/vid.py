import numpy as np
import keras
import tensorflow as tf


# based on the implementation in https://github.com/sseung0703/KD_methods_with_TF
# and https://github.com/ssahn0215/variational-information-distillation
def vid_loss(target, inputs):
    channel_num = inputs.get_shape().as_list()[-1]
    alpha = tf.get_variable('alpha_'+inputs.name.split('/')[0], [1,1,1,channel_num], tf.float32, trainable=True, initializer=tf.constant_initializer(5.0))
    var = tf.math.softplus(alpha) + 1.0

    # use 1x1 conv layer as W here, due to the same structure of teacher and student models
    x = keras.layers.Conv2D(channel_num, kernel_size=1, strides=1, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization(center=False, scale=False)(x)
    mean_s = keras.layers.Activation('relu')(x)

    loss = tf.reduce_mean(tf.log(var) + tf.square(target - mean_s) / var) / 2.0 
    return loss





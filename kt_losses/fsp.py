import numpy as np
import tensorflow as tf


def gram_fsp_resnet(conn):
    x = conn[0]
    y = conn[1]
    
    # here we directly cut a range from input tensor on height axis for simplicity
    x = tf.nn.max_pool(x, [1,1,2,1], [1,1,2,1], padding='SAME')
    y = y[:, :tf.shape(x)[1], :, :]
    x = x[:, :, :tf.shape(y)[2], :]
    x = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[3]])
    y = tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[3]])
    res = tf.matmul(x, y, transpose_a=True) / tf.cast(tf.shape(x)[1], dtype=tf.float32)

    return res

def gram_fsp_fcnn(conn):
    x = conn[0]
    y = conn[1]
    
    x = tf.nn.max_pool(x, [1,2,1,1], [1,2,1,1], padding='SAME')
    y = y[:, :tf.shape(x)[1], :, :]
    x = x[:, :, :tf.shape(y)[2], :]
    x = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[3]])
    y = tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[3]])
    res = tf.matmul(x, y, transpose_a=True) / tf.cast(tf.shape(x)[1], dtype=tf.float32)

    return res

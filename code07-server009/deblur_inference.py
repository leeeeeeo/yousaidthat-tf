import tensorflow as tf
import numpy as np


config_dict = {
    'supr0': [[3, 3, 3, 64], [64]],
    'supr1': [[3, 3, 64, 64], [64]],
    'supr2': [[3, 3, 64, 64], [64]],
    'supr3': [[3, 3, 64, 64], [64]],
    'supr4': [[3, 3, 64, 64], [64]],
    'supr5': [[3, 3, 64, 64], [64]],
    'supr6': [[3, 3, 64, 64], [64]],
    'supr7': [[3, 3, 64, 64], [64]],
    'supr8': [[3, 3, 64, 64], [64]],
    'supr9': [[3, 3, 64, 3], [64]],
}


def inference(input_lip):
    print 'build model started'

    supr0 = conv_layer(input_lip, 'supr0')
    resr0 = relu_layer(supr0, 'resr0')
    supr1 = conv_layer(resr0, 'supr1')
    resr1 = relu_layer(supr1, 'resr1')
    supr2 = conv_layer(resr1, 'supr2')
    resr2 = relu_layer(supr2, 'resr2')
    supr3 = conv_layer(resr2, 'supr3')
    resr3 = relu_layer(supr3, 'resr3')
    supr4 = conv_layer(resr3, 'supr4')
    resr4 = relu_layer(supr4, 'resr4')
    supr5 = conv_layer(resr4, 'supr5')
    resr5 = relu_layer(supr5, 'resr5')
    supr6 = conv_layer(resr5, 'supr6')
    resr6 = relu_layer(supr6, 'resr6')
    supr7 = conv_layer(resr6, 'supr7')
    resr7 = relu_layer(supr7, 'resr7')
    supr8 = conv_layer(resr7, 'supr8')
    resr8 = relu_layer(supr8, 'resr8')
    supr9 = conv_layer(resr8, 'supr9')

    return supr9


def conv_layer(bottom, name):
    with tf.variable_scope(name):
        biases = tf.get_variable(
            "bias", config_dict[name][1], initializer=tf.constant_initializer(0.0))
        weights = tf.get_variable(
            "weight", config_dict[name][0], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        return tf.nn.bias_add(conv, biases)


def relu_layer(bottom, name):
    with tf.variable_scope(name):
        return tf.nn.relu(bottom)

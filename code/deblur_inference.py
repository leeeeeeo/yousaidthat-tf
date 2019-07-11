import tensorflow as tf
import numpy as np
import deblur_utils
import scipy.io as scio


class Deblur:
    def __init__(self, USE_PRETRAINED_CONV):
        self.config_dict = deblur_utils.create_config_dict()
        self.v114_param_dict, self.v114_param_list = self.read_v114_param(
            USE_PRETRAINED_CONV)

    def inference(self, input_lip):
        print '============ BUILD DEBLUR INFERENCE ============'
        supr0 = self.conv_layer(input_lip, 'supr0')
        resr0 = self.relu_layer(supr0, 'resr0')
        supr1 = self.conv_layer(resr0, 'supr1')
        resr1 = self.relu_layer(supr1, 'resr1')
        supr2 = self.conv_layer(resr1, 'supr2')
        resr2 = self.relu_layer(supr2, 'resr2')
        supr3 = self.conv_layer(resr2, 'supr3')
        resr3 = self.relu_layer(supr3, 'resr3')
        supr4 = self.conv_layer(resr3, 'supr4')
        resr4 = self.relu_layer(supr4, 'resr4')
        supr5 = self.conv_layer(resr4, 'supr5')
        resr5 = self.relu_layer(supr5, 'resr5')
        supr6 = self.conv_layer(resr5, 'supr6')
        resr6 = self.relu_layer(supr6, 'resr6')
        supr7 = self.conv_layer(resr6, 'supr7')
        resr7 = self.relu_layer(supr7, 'resr7')
        supr8 = self.conv_layer(resr7, 'supr8')
        resr8 = self.relu_layer(supr8, 'resr8')
        supr9 = self.conv_layer(resr8, 'supr9')
        return supr9

    def compute_loss(self, prediction, groundtruth, loss_func):
        if loss_func == 'mse':
            return tf.losses.mean_squared_error(groundtruth, prediction)
        elif loss_func == 'l2':
            return tf.nn.l2_loss(prediction-groundtruth)

    def relu_layer(self, bottom, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            return tf.nn.relu(bottom)

    def conv_layer(self, bottom, name):
        padding_mode = self.config_dict[name][2]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if name in self.v114_param_list:
                print '{} using pre-trained parameter'.format(name)
                filt_key = '{}.f'.format(name)
                filt_init = tf.constant_initializer(
                    self.v114_param_dict[filt_key])
                filt = tf.get_variable(
                    'filter', self.config_dict[name][0], initializer=filt_init)
                conv_biases_key = '{}.b'.format(name)
                conv_biases_init = tf.constant_initializer(
                    self.v114_param_dict[conv_biases_key])
                conv_biases = tf.get_variable(
                    'biases', self.config_dict[name][3], initializer=conv_biases_init)
            else:
                print '{} not using pre-trained parameter'.format(name)
                filt = tf.get_variable('filter', self.config_dict[name][0], initializer=tf.truncated_normal_initializer(
                    stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
                conv_biases = tf.get_variable(
                    'biases', self.config_dict[name][3], dtype=tf.float32)
            strides = self.config_dict[name][1]
            paddings = tf.constant([[0, 0], [padding_mode[0], padding_mode[1]], [
                                    padding_mode[2], padding_mode[3]], [0, 0]])
            bottom = tf.pad(bottom, paddings, "CONSTANT")
            conv = tf.nn.conv2d(
                bottom, filt, strides=strides, padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

    def read_v114_param(self, USE_PRETRAINED_CONV):
        # server009
        v114_param_path = '/media/server009/seagate/liuhan/lls/github/yousaidthat/v114_param_cell.mat'
        # mac
        # v114_param_path = '/Users/lls/Documents/face/github/yousaidthat/v114_param_cell.mat'
        # ssh
        # v114_param_path='/workspace/liuhan/work/avasyn/github/yousaidthat/v114_param_cell.mat'
        v114_param = scio.loadmat(v114_param_path)
        v114_param_dict = {}
        v114_param_list = []
        for i in range(0, v114_param['v114_param_cell'].shape[1]):
            key = str(v114_param['v114_param_cell'][0][i][0])
            value = v114_param['v114_param_cell'][1][i]
            v114_param_dict[key] = value
            if not key.split('.')[0] in v114_param_list:
                v114_param_list.append(key.split('.')[0])
        if USE_PRETRAINED_CONV == 0:
            v114_param_list = []
        return v114_param_dict, v114_param_list

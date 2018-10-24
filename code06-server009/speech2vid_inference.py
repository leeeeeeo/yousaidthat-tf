# -*- coding:utf-8 -*-
# NHWC
# 修改所有的padding

import tensorflow as tf
import numpy as np
import types
import speech2vid_utils


BN_EPSILON = 0.0001

class Speech2Vid:
    def __init__(self, checkpoint_dir=None):
        self.config_dict = speech2vid_utils.create_config_dict()
        print 'config loaded'
        if checkpoint_dir is None:
            checkpoint_dir = './'

    def audio_encoder(self,input_audio):
        print 'build audio encoder'
        x1_audio = self.conv_layer(input_audio, 'conv1_audio')
        x2_audio = self.bn_layer(x1_audio, 'bn1_audio')
        x3_audio = self.relu_layer(x2_audio, 'relu1_audio')
        x4_audio = self.max_pool_layer(x3_audio, 'pool1_audio')
        x5_audio = self.conv_layer(x4_audio, 'conv2_audio')
        x6_audio = self.bn_layer(x5_audio, 'bn2_audio')
        x7_audio = self.relu_layer(x6_audio, 'relu2_audio')
        x8_audio = self.max_pool_layer(x7_audio, 'pool2_audio')
        x9_audio = self.conv_layer(x8_audio, 'conv3_audio')
        x10_audio = self.bn_layer(x9_audio, 'bn3_audio')
        x11_audio = self.relu_layer(x10_audio, 'relu3_audio')
        x12_audio = self.conv_layer(x11_audio, 'conv4_audio')
        x13_audio = self.bn_layer(x12_audio, 'bn4_audio')
        x14_audio = self.relu_layer(x13_audio, 'relu4_audio')
        x15_audio = self.conv_layer(x14_audio, 'conv5_audio')
        x16_audio = self.bn_layer(x15_audio, 'bn5_audio')
        x17_audio = self.relu_layer(x16_audio, 'relu5_audio')
        x18_audio = self.max_pool_layer(x17_audio, 'pool5_audio')
        # x19_audio = self.fc_layer(x18_audio, 'fc6_audio')
        x19_audio = self.conv_layer(x18_audio, 'fc6_audio')
        x20_audio = self.bn_layer(x19_audio, 'bn6_audio')
        x21_audio = self.relu_layer(x20_audio, 'relu6_audio')
        # x22_audio = self.fc_layer(x21_audio, 'fc7_audio')
        x22_audio = self.conv_layer(x21_audio, 'fc7_audio')
        x23_audio = self.bn_layer(x22_audio, 'bn7_audio')
        x24_audio = self.relu_layer(x23_audio, 'relu7_audio')
        return x24_audio

    def identity_encoder(self,input_face):
        print 'build identity encoder'
        x1_face = self.conv_layer(input_face, 'conv1_face')
        x2_face = self.bn_layer(x1_face, 'bn1_face')
        x3_face = self.relu_layer(x2_face, 'relu1_face')
        x4_face = self.max_pool_layer(x3_face, 'pool1_face')
        x5_face = self.conv_layer(x4_face, 'conv2_face')
        x6_face = self.bn_layer(x5_face, 'bn2_face')
        x7_face = self.relu_layer(x6_face, 'relu2_face')
        x8_face = self.max_pool_layer(x7_face, 'pool2_face')
        x9_face = self.conv_layer(x8_face, 'conv3_face')
        x10_face = self.bn_layer(x9_face, 'bn3_face')
        x11_face = self.relu_layer(x10_face, 'relu3_face')
        x12_face = self.conv_layer(x11_face, 'conv4_face')
        x13_face = self.bn_layer(x12_face, 'bn4_face')
        x14_face = self.relu_layer(x13_face, 'relu4_face')
        x15_face = self.conv_layer(x14_face, 'conv5_face')
        x16_face = self.bn_layer(x15_face, 'bn5_face')
        x17_face = self.relu_layer(x16_face, 'relu5_face')
        x18_face = self.conv_layer(x17_face, 'conv6_face')
        x19_face = self.bn_layer(x18_face, 'bn6_face')
        x20_face = self.relu_layer(x19_face, 'relu6_face')
        x21_face = self.conv_layer(x20_face, 'conv7_face')
        x22_face = self.bn_layer(x21_face, 'bn7_face')
        x23_face = self.relu_layer(x22_face, 'relu7_face')
        return x23_face,x7_face,x4_face,x3_face

    def image_decoder(self,input_audio,input_face):
        print 'build image decoder'
        audio_encoder_output=self.audio_encoder(input_audio)
        identity_encoder_output,x7_face,x4_face,x3_face=self.identity_encoder(input_face)
        x24 = self.concat_layer(audio_encoder_output, identity_encoder_output, 'concat8')
        x26 = self.conv_layer(x24, 'conv8')
        relu8 = self.relu_layer(x26, 'relu8')
        upsamp1_1 = self.upsample_layer(relu8, 'upsamp1_1')
        conv1_1 = self.conv_layer(upsamp1_1, 'conv1_1')
        relu1_1 = self.relu_layer(conv1_1, 'relu1_1')
        upsamp1_2 = self.upsample_layer(relu1_1, 'upsamp1_2')
        conv1_2 = self.conv_layer(upsamp1_2, 'conv1_2')
        relu1_2 = self.relu_layer(conv1_2, 'relu1_2')
        upsamp1_3 = self.upsample_layer(relu1_2, 'upsamp1_3')
        conv1_3 = self.conv_layer(upsamp1_3, 'conv1_3')
        relu1_3 = self.relu_layer(conv1_3, 'relu1_3')
        upsamp2 = self.upsample_layer(relu1_3, 'upsamp2')
        conv2 = self.conv_layer(upsamp2, 'conv2')
        relu2 = self.relu_layer(conv2, 'relu2')
        skipCat2 = self.concat_layer(x7_face, relu2, 'concatSkip2')
        upsamp3_1 = self.upsample_layer(skipCat2, 'upsamp3_1')
        conv3_1 = self.conv_layer(upsamp3_1, 'conv3_1')
        relu3_1 = self.relu_layer(conv3_1, 'relu3_1')
        conv3_2 = self.conv_layer(relu3_1, 'conv3_2')
        relu3_2 = self.relu_layer(conv3_2, 'relu3_2')
        skipCat3 = self.concat_layer(x4_face, relu3_2, 'concatSkip3')
        upsamp4 = self.upsample_layer(skipCat3, 'upsamp4')
        conv4 = self.conv_layer(upsamp4, 'conv4')
        relu4 = self.relu_layer(conv4, 'relu4')
        skipCat4 = self.concat_layer(x3_face, relu4, 'concatSkip4')
        upsamp5_1 = self.upsample_layer(skipCat4, 'upsamp5_1')
        conv5_1 = self.conv_layer(upsamp5_1, 'conv5_1')
        relu5_1 = self.relu_layer(conv5_1, 'relu5_1')
        conv5_2 = self.conv_layer(relu5_1, 'conv5_2')
        relu5_2 = self.relu_layer(conv5_2, 'relu5_2')
        conv5_3 = self.conv_layer(relu5_2, 'conv5_3')
        prediction = self.sigmoid_layer(conv5_3, 'sigmoid5_3')
        return prediction

    def loss_encoder(self,input_image):
        x1_face_lip_1 = self.conv_layer(input_image, 'conv1_face_lip_1')
        x2_face_lip_1 = self.bn_layer(x1_face_lip_1, 'bn1_face_lip_1')
        x3_face_lip_1 = self.relu_layer(x2_face_lip_1, 'relu1_face_lip_1')
        x4_face_lip_1 = self.max_pool_layer(x3_face_lip_1, 'pool1_face_lip_1')
        x5_face_lip_1 = self.conv_layer(x4_face_lip_1, 'conv2_face_lip_1')
        x6_face_lip_1=self.bn_layer(x5_face_lip_1,'bn2_face_lip_1')
        x7_face_lip_1=self.relu_layer(x6_face_lip_1,'relu2_face_lip_1')
        x8_face_lip_1=self.max_pool_layer(x7_face_lip_1,'pool2_face_lip_1')
        x9_face_lip_1=self.conv_layer(x8_face_lip_1,'conv3_face_lip_1')
        x10_face_lip_1=self.bn_layer(x9_face_lip_1,'bn3_face_lip_1')
        x11_face_lip_1=self.relu_layer(x10_face_lip_1,'relu3_face_lip_1')
        x12_face_lip_1=self.conv_layer(x11_face_lip_1,'conv4_face_lip_1')
        x13_face_lip_1=self.bn_layer(x12_face_lip_1,'bn4_face_lip_1')
        x14_face_lip_1=self.relu_layer(x13_face_lip_1,'relu4_face_lip_1')
        x15_face_lip_1=self.conv_layer(x14_face_lip_1,'conv5_face_lip_1')
        x16_face_lip_1=self.bn_layer(x15_face_lip_1,'bn5_face_lip_1')
        x17_face_lip_1=self.relu_layer(x16_face_lip_1,'relu5_face_lip_1')
        return x3_face_lip_1,x7_face_lip_1,x11_face_lip_1,x14_face_lip_1,x17_face_lip_1


    def inference(self,input_audio,input_face,batch_size):
        prediction=self.image_decoder(input_audio,input_face)
        return prediction


    def compute_loss(self,prediction,face_gt_batch):
        groundtruth=face_gt_batch
        x3_face_lip_1,x7_face_lip_1,x11_face_lip_1,x14_face_lip_1,x17_face_lip_1=self.loss_encoder(prediction)
        x3_face_lip_2,x7_face_lip_2,x11_face_lip_2,x14_face_lip_2,x17_face_lip_2=self.loss_encoder(groundtruth)
        # loss0: prediction, input
        loss0=tf.reduce_mean(tf.abs(prediction-groundtruth))
        # loss1: x3_face_lip_1, x3_face_lip_2
        loss1=tf.reduce_mean(tf.abs(x3_face_lip_1-x3_face_lip_2))
        # loss2: x7_face_lip_1, x7_face_lip_2
        loss2=tf.reduce_mean(tf.abs(x7_face_lip_1-x7_face_lip_2))
        # loss3: x11_face_lip_1, x11_face_lip_2
        loss3=tf.reduce_mean(tf.abs(x11_face_lip_1-x11_face_lip_2))
        # loss4: x14_face_lip_1, x14_face_lip_1
        loss4=tf.reduce_mean(tf.abs(x14_face_lip_1-x14_face_lip_2))
        # loss4: x17_face_lip_1, x17_face_lip_2
        loss5=tf.reduce_mean(tf.abs(x17_face_lip_1-x17_face_lip_2))
        # ?????
        loss=loss0+loss1+loss2+loss3+loss4+loss5
        return loss







    def sigmoid_layer(self, bottom, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            return tf.nn.sigmoid(bottom)

    def upsample_layer(self, bottom, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            old_height = bottom.shape[1]
            old_width = bottom.shape[2]
            scale = 2
            new_height = old_height*scale
            new_width = old_width*scale
            return tf.image.resize_images(bottom, [new_height, new_width])

    def concat_layer(self, bottom1, bottom2, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            #NHWC
            return tf.concat(values=[bottom1, bottom2], axis=3,name=name)



    def fc_layer(self,bottom,name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            fc=tf.layers.dense(inputs=bottom,units=self.config_dict[name][3][0],activation=None,use_bias=True,kernel_initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32))
            return fc
    # def fc_layer(self, bottom, name):
    #     print bottom.shape
    #     with tf.variable_scope(name):
    #         shape = bottom.get_shape().as_list()
    #         dim = 1
    #         for d in shape[1:]:
    #             dim *= d
    #         x = tf.reshape(bottom, [-1, dim])
    #         # weights = tf.constant(self.config_dict[name][0], name="weights",dtype=tf.float32)
    #         # biases = tf.constant(self.config_dict[name][3], name="biases",dtype=tf.float32)
    #         weights = tf.get_variable('weights',self.config_dict[name][0],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32),dtype=tf.float32)
    #         biases = tf.get_variable('biases',self.config_dict[name][3],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32),dtype=tf.float32)
    #         print weights.shape,biases.shape
    #         fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
    #         return fc

    def max_pool_layer(self, bottom, name):
        padding_mode=self.config_dict[name][2]
        if padding_mode=='valid':
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                # ksize = tf.constant(self.config_dict[name][0], name='ksize')
                ksize=self.config_dict[name][0]
                strides = self.config_dict[name][1]
                return tf.layers.max_pooling2d(bottom,pool_size=ksize,strides=strides,padding='VALID',name=name)
        elif isinstance(padding_mode,list):
            if len(padding_mode)==2:
                with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                    ksize = self.config_dict[name][0]
                    strides = self.config_dict[name][1]
                    paddings=tf.constant([[0, 0], [padding_mode[0], padding_mode[0]], [padding_mode[1], padding_mode[1]], [0, 0]])
                    bottom=tf.pad(bottom,paddings,"CONSTANT")
                    return tf.layers.max_pooling2d(bottom,pool_size=ksize,strides=strides,padding='VALID',name=name)
            elif len(padding_mode)==4:
                with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                    ksize = self.config_dict[name][0]
                    strides = self.config_dict[name][1]

                    paddings=tf.constant([[0, 0], [padding_mode[0], padding_mode[1]], [padding_mode[2], padding_mode[3]], [0, 0]])
                    bottom=tf.pad(bottom,paddings,"CONSTANT")
                    # top_padding=np.zeros((bottom.shape[0],padding_mode[0],bottom.shape[2],bottom.shape[3]))
                    # bottom_padding=np.zeros((bottom.shape[0],padding_mode[1],bottom.shape[2],bottom.shape[3]))
                    # left_padding=np.zeros((bottom.shape[0],bottom.shape[1],padding_mode[2],bottom.shape[3]))
                    # right_padding=np.zeros((bottom.shape[0],bottom.shape[1],padding_mode[3],bottom.shape[3]))

                    # bottom=tf.concat([top_padding,bottom],1)
                    # bottom=tf.concat([bottom,bottom_padding],1)
                    # bottom=tf.concat([left_padding,bottom],2)
                    # bottom=tf.concat([bottom,right_padding],2)
                
                    return tf.layers.max_pooling2d(bottom,pool_size=ksize,strides=strides,padding='VALID',name=name)

    def relu_layer(self, bottom, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            return tf.nn.relu(bottom)

    def conv_layer(self, bottom, name):
        padding_mode=self.config_dict[name][2]
        if padding_mode=='valid':
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE) :
                filt=tf.get_variable('filter',self.config_dict[name][0],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32),dtype=tf.float32)
                strides=self.config_dict[name][1]
                conv = tf.nn.conv2d(bottom, filt, strides=strides, padding='VALID')
                conv_biases = tf.get_variable('biases',self.config_dict[name][3],dtype=tf.float32)
                bias = tf.nn.bias_add(conv, conv_biases)
                return bias
        elif isinstance(padding_mode,list):
            if len(padding_mode)==2:
                with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                    # filt = tf.constant(self.config_dict[name][0], name='filter',dtype=tf.float32)
                    filt=tf.get_variable('filter',self.config_dict[name][0],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32),dtype=tf.float32)
                    strides=self.config_dict[name][1]
                    paddings=tf.constant([[0, 0], [padding_mode[0], padding_mode[0]], [padding_mode[1], padding_mode[1]], [0, 0]])
                    bottom=tf.pad(bottom,paddings,"CONSTANT")
                    conv = tf.nn.conv2d(bottom, filt, strides=strides, padding='VALID')
                    # conv_biases = tf.constant(self.config_dict[name][3], name='biases',dtype=tf.float32)
                    conv_biases=tf.get_variable('biases',self.config_dict[name][3],dtype=tf.float32)
                    bias = tf.nn.bias_add(conv, conv_biases)
                    return bias
            elif len(padding_mode)==4:
                with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                    # new
                    filt=tf.get_variable('filter',self.config_dict[name][0],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32),dtype=tf.float32)
                    strides=self.config_dict[name][1]
                    paddings=tf.constant([[0, 0], [padding_mode[0], padding_mode[1]], [padding_mode[2], padding_mode[3]], [0, 0]])
                    bottom=tf.pad(bottom,paddings,"CONSTANT")
                    conv = tf.nn.conv2d(bottom, filt, strides=strides, padding='VALID')
                    conv_biases=tf.get_variable('biases',self.config_dict[name][3],dtype=tf.float32)
                    bias = tf.nn.bias_add(conv, conv_biases)
                    return bias





    def bn_layer(self, bottom, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            #NHWC axis=batch(N)=0
            return tf.layers.batch_normalization(bottom,axis=0)

















    # def build(self, input_audio, input_face, input_lip):
    #     print 'build model started'
    #     self.x1_audio = self.conv_layer(input_audio, 'conv1_audio')
    #     self.x2_audio = self.bn_layer(self.x1_audio, 'bn1_audio')
    #     self.x3_audio = self.relu_layer(self.x2_audio, 'relu1_audio')
    #     self.x4_audio = self.max_pool_layer(self.x3_audio, 'pool1_audio')
    #     self.x5_audio = self.conv_layer(self.x4_audio, 'conv2_audio')
    #     self.x6_audio = self.bn_layer(self.x5_audio, 'bn2_audio')
    #     self.x7_audio = self.relu_layer(self.x6_audio, 'relu2_audio')
    #     self.x8_audio = self.max_pool_layer(self.x7_audio, 'pool2_audio')
    #     self.x9_audio = self.conv_layer(self.x8_audio, 'conv3_audio')
    #     self.x10_audio = self.bn_layer(self.x9_audio, 'bn3_audio')
    #     self.x11_audio = self.relu_layer(self.x10_audio, 'relu3_audio')
    #     self.x12_audio = self.conv_layer(self.x11_audio, 'conv4_audio')
    #     self.x13_audio = self.bn_layer(self.x12_audio, 'bn4_audio')
    #     self.x14_audio = self.relu_layer(self.x13_audio, 'relu4_audio')
    #     self.x15_audio = self.conv_layer(self.x14_audio, 'conv5_audio')
    #     self.x16_audio = self.bn_layer(self.x15_audio, 'bn5_audio')
    #     self.x17_audio = self.relu_layer(self.x16_audio, 'relu5_audio')
    #     self.x18_audio = self.max_pool_layer(self.x17_audio, 'pool5_audio')
    #     self.x19_audio = self.fc_layer(self.x18_audio, 'fc6_audio')
    #     self.x20_audio = self.bn_layer(self.x19_audio, 'bn6_audio')
    #     self.x21_audio = self.relu_layer(self.x20_audio, 'relu6_audio')
    #     self.x22_audio = self.fc_layer(self.x21_audio, 'fc7_audio')
    #     self.x23_audio = self.bn_layer(self.x22_audio, 'bn7_audio')
    #     self.x24_audio = self.relu_layer(self.x23_audio, 'relu7_audio')

    #     self.x1_face = self.conv_layer(input_face, 'conv1_face')
    #     self.x2_face = self.bn_layer(self.x1_face, 'bn1_face')
    #     self.x3_face = self.relu_layer(self.x2_face, 'relu1_face')
    #     self.x4_face = self.max_pool_layer(self.x3_face, 'pool1_face')
    #     self.x5_face = self.conv_layer(self.x4_face, 'conv2_face')
    #     self.x6_face = self.bn_layer(self.x5_face, 'bn2_face')
    #     self.x7_face = self.relu_layer(self.x6_face, 'relu2_face')
    #     self.x8_face = self.max_pool_layer(self.x7_face, 'pool2_face')
    #     self.x9_face = self.conv_layer(self.x8_face, 'conv3_face')
    #     self.x10_face = self.bn_layer(self.x9_face, 'bn3_face')
    #     self.x11_face = self.relu_layer(self.x10_face, 'relu3_face')
    #     self.x12_face = self.conv_layer(self.x11_face, 'conv4_face')
    #     self.x13_face = self.bn_layer(self.x12_face, 'bn4_face')
    #     self.x14_face = self.relu_layer(self.x13_face, 'relu4_face')
    #     self.x15_face = self.conv_layer(self.x14_face, 'conv5_face')
    #     self.x16_face = self.bn_layer(self.x15_face, 'bn5_face')
    #     self.x17_face = self.relu_layer(self.x16_face, 'relu5_face')
    #     self.x18_face = self.conv_layer(self.x17_face, 'conv6_face')
    #     self.x19_face = self.bn_layer(self.x18_face, 'bn6_face')
    #     self.x20_face = self.relu_layer(self.x19_face, 'relu6_face')
    #     self.x21_face = self.conv_layer(self.x20_face, 'conv7_face')
    #     self.x22_face = self.bn_layer(self.x21_face, 'bn7_face')
    #     self.x23_face = self.relu_layer(self.x22_face, 'relu7_face')

    #     self.x24 = self.concat_layer(self.x24_audio, self.x23_face, 'concat8')
    #     self.x26 = self.conv_layer(self.x24, 'conv8')
    #     self.relu8 = self.relu_layer(self.x26, 'relu8')
    #     self.upsamp1_1 = self.upsample_layer(self.relu8, 'upsamp1_1')
    #     self.conv1_1 = self.conv_layer(self.upsamp1_1, 'conv1_1')
    #     self.relu1_1 = self.relu_layer(self.conv1_1, 'relu1_1')
    #     self.upsamp1_2 = self.upsample_layer(self.relu1_1, 'upsamp1_2')
    #     self.conv1_2 = self.conv_layer(self.upsamp1_2, 'conv1_2')
    #     self.relu1_2 = self.relu_layer(self.conv1_2, 'relu1_2')
    #     self.upsamp1_3 = self.upsample_layer(self.relu1_2, 'upsamp1_3')
    #     self.conv1_3 = self.conv_layer(self.upsamp1_3, 'conv1_3')
    #     self.relu1_3 = self.relu_layer(self.conv1_3, 'relu1_3')
    #     self.upsamp2 = self.upsample_layer(self.relu1_3, 'upsamp2')
    #     self.conv2 = self.conv_layer(self.upsamp2, 'conv2')
    #     self.relu2 = self.relu_layer(self.conv2, 'relu2')
    #     self.skipCat2 = self.concat_layer(
    #         self.x7_face, self.relu2, 'concatSkip2')
    #     self.upsamp3_1 = self.upsample_layer(self.skipCat2, 'upsamp3_1')
    #     self.conv3_1 = self.conv_layer(self.upsamp3_1, 'conv3_1')
    #     self.relu3_1 = self.relu_layer(self.conv3_1, 'relu3_1')
    #     self.conv3_2 = self.conv_layer(self.relu3_1, 'conv3_2')
    #     self.relu3_2 = self.relu_layer(self.conv3_2, 'relu3_2')
    #     self.skipCat3 = self.concat_layer(
    #         self.x4_face, self.relu3_2, 'concatSkip3')
    #     self.upsamp4 = self.upsample_layer(self.skipCat3, 'upsamp4')
    #     self.conv4 = self.conv_layer(self.upsamp4, 'conv4')
    #     self.relu4 = self.relu_layer(self.conv4, 'relu4')
    #     self.skipCat4 = self.concat_layer(
    #         self.x3_face, self.relu4, 'concatSkip4')
    #     self.upsamp5_1 = self.upsample_layer(self.skipCat4, 'upsamp5_1')
    #     self.conv5_1 = self.conv_layer(self.upsamp5_1, 'conv5_1')
    #     self.relu5_1 = self.relu_layer(self.conv5_1, 'relu5_1')
    #     self.conv5_2 = self.conv_layer(self.relu5_1, 'conv5_2')
    #     self.relu5_2 = self.relu_layer(self.conv5_2, 'relu5_2')
    #     self.conv5_3 = self.conv_layer(self.relu5_2, 'conv5_3')
    #     self.prediction = self.sigmoid_layer(self.conv5_3, 'sigmoid5_3')

    #     self.x1_face_lip_1 = self.conv_layer(
    #         self.prediction, 'conv1_face_lip_1')
    #     self.x1_face_lip_2 = self.conv_layer(
    #         self.input_lip, 'conv1_face_lip_2')
    #     self.x2_face_lip_1 = self.bn_layer(
    #         self.x1_face_lip_1, 'bn1_face_lip_1')
    #     self.x2_face_lip_2 = self.bn_layer(
    #         self.x1_face_lip_2, 'bn1_face_lip_2')
    #     self.x3_face_lip_1 = self.relu_layer(
    #         self.x2_face_lip_1, 'relu1_face_lip_1')
    #     self.x3_face_lip_2 = self.relu_layer(
    #         self.x2_face_lip_2, 'relu1_face_lip_2')
    #     self.x4_face_lip_1 = self.max_pool_layer(
    #         self.x3_face_lip_1, 'pool1_face_lip_1')
    #     self.x4_face_lip_2 = self.max_pool_layer(
    #         self.x3_face_lip_2, 'pool1_face_lip_2')
    #     self.x5_face_lip_1 = self.conv_layer(
    #         self.x4_face_lip_1, 'conv2_face_lip_1')
    #     self.x5_face_lip_2 = self.conv_layer(
    #         self.x4_face_lip_2, 'conv2_face_lip_2')
    #     self.x6_face_lip_1=self.bn_layer(self.x5_face_lip_1,'bn2_face_lip_1')
    #     self.x6_face_lip_2=self.bn_layer(self.x5_face_lip_2,'bn2_face_lip_2')
    #     self.x7_face_lip_1=self.relu_layer(self.x6_face_lip_1,'relu2_face_lip_1')
    #     self.x7_face_lip_2=self.relu_layer(self.x6_face_lip_2,'relu2_face_lip_2')
    #     self.x8_face_lip_1=self.max_pool_layer(self.x7_face_lip_1,'pool2_face_lip_1')
    #     self.x8_face_lip_2=self.max_pool_layer(self.x7_face_lip_2,'pool2_face_lip_2')
    #     self.x9_face_lip_1=self.conv_layer(self.x8_face_lip_1,'conv3_face_lip_1')
    #     self.x9_face_lip_2=self.conv_layer(self.x8_face_lip_2,'conv3_face_lip_2')
    #     self.x10_face_lip_1=self.bn_layer(self.x9_face_lip_1,'bn3_face_lip_1')
    #     self.x10_face_lip_2=self.bn_layer(self.x9_face_lip_2,'bn3_face_lip_2')
    #     self.x11_face_lip_1=self.relu_layer(self.x10_face_lip_1,'relu3_face_lip_1')
    #     self.x11_face_lip_2=self.relu_layer(self.x10_face_lip_2,'relu3_face_lip_2')
    #     self.x12_face_lip_1=self.conv_layer(self.x11_face_lip_1,'conv4_face_lip_1')
    #     self.x12_face_lip_2=self.conv_layer(self.x11_face_lip_2,'conv4_face_lip_2')
    #     self.x13_face_lip_1=self.bn_layer(self.x12_face_lip_1,'bn4_face_lip_1')
    #     self.x13_face_lip_2=self.bn_layer(self.x12_face_lip_2,'bn4_face_lip_2')
    #     self.x14_face_lip_1=self.relu_layer(self.x13_face_lip_1,'relu4_face_lip_1')
    #     self.x14_face_lip_2=self.relu_layer(self.x13_face_lip_2,'relu4_face_lip_2')
    #     self.x15_face_lip_1=self.conv_layer(self.x14_face_lip_1,'conv5_face_lip_1')
    #     self.x15_face_lip_2=self.conv_layer(self.x14_face_lip_2,'conv5_face_lip_2')
    #     self.x16_face_lip_1=self.bn_layer(self.x15_face_lip_1,'bn5_face_lip_1')
    #     self.x16_face_lip_2=self.bn_layer(self.x15_face_lip_2,'bn5_face_lip_2')
    #     self.x17_face_lip_1=self.relu_layer(self.x16_face_lip_1,'relu5_face_lip_1')
    #     self.x17_face_lip_2=self.relu_layer(self.x16_face_lip_2,'relu5_face_lip_2')
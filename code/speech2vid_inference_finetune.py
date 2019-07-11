# -*- coding:utf-8 -*-
# NHWC


import tensorflow as tf
import numpy as np
import types
import speech2vid_utils
import scipy.io as scio




class Speech2Vid:
    
    def __init__(self, USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER,AVOID_LAYERS_LIST,IDNUM):
        self.USE_XAVIER=USE_XAVIER
        self.config_dict = speech2vid_utils.create_config_dict()
        self.v201_param_dict,self.v201_param_list=self.read_v201_param(USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC,USE_AUDIOFC,USE_FACE,AVOID_LAYERS_LIST,IDNUM)

    def audio_encoder(self,input_audio):
        print '============ BUILD AUDIO ENCODER ============'
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

    def identity_encoder(self,input_face,idnum):
        print '============ BUILD IDENTITY ENCODER ============'
        if idnum==1:
            tf.summary.image("identity1", input_face)
            x1_face = self.conv_layer(input_face, 'conv1_face_idnum1')
        elif idnum == 5:
            tf.summary.image("identity1",tf.split(input_face,5,-1)[0])
            tf.summary.image("identity2",tf.split(input_face,5,-1)[1])
            tf.summary.image("identity3",tf.split(input_face,5,-1)[2])
            tf.summary.image("identity4",tf.split(input_face,5,-1)[3])
            tf.summary.image("identity5", tf.split(input_face, 5, -1)[4])
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

    def image_decoder(self,audio_encoder_output,identity_encoder_output,x7_face,x4_face,x3_face):
        print '============ BUILD DECODER ============'
        # old
        x24 = self.concat_layer(audio_encoder_output, identity_encoder_output, 'concat8')
        # new
        # x24 = self.concat_layer(identity_encoder_output, audio_encoder_output, 'concat8')
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
        # tf.summary.image("conv5_3",conv5_3)
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


    def inference(self,input_audio,input_face):
        audio_encoder_output=self.audio_encoder(input_audio)
        identity_encoder_output,x7_face,x4_face,x3_face=self.identity_encoder(input_face)
        prediction=self.image_decoder(audio_encoder_output,identity_encoder_output,x7_face,x4_face,x3_face)
        return prediction


    def loss_l1(self, prediction, face_gt_batch):
        tf.summary.image("prediction_when_compute_loss",  prediction)
        tf.summary.image("groundtruth_when_compute_loss",  face_gt_batch)
        x3_face_lip_1,x7_face_lip_1,x11_face_lip_1,x14_face_lip_1,x17_face_lip_1=self.loss_encoder(prediction)
        x3_face_lip_2,x7_face_lip_2,x11_face_lip_2,x14_face_lip_2,x17_face_lip_2=self.loss_encoder(face_gt_batch)
        # loss0: prediction, input
        loss0=tf.losses.absolute_difference(face_gt_batch,prediction)
        # loss0=tf.reduce_mean(tf.abs(prediction-face_gt_batch))
        # loss1: x3_face_lip_1, x3_face_lip_2
        loss1=tf.losses.absolute_difference(x3_face_lip_2,x3_face_lip_1)
        # loss1=tf.reduce_mean(tf.abs(x3_face_lip_1-x3_face_lip_2))
        # loss2: x7_face_lip_1, x7_face_lip_2
        loss2=tf.losses.absolute_difference(x7_face_lip_2,x7_face_lip_1)
        # loss2=tf.reduce_mean(tf.abs(x7_face_lip_1-x7_face_lip_2))
        # loss3: x11_face_lip_1, x11_face_lip_2
        loss3=tf.losses.absolute_difference(x11_face_lip_2,x11_face_lip_1)
        # loss3=tf.reduce_mean(tf.abs(x11_face_lip_1-x11_face_lip_2))
        # loss4: x14_face_lip_1, x14_face_lip_1
        loss4=tf.losses.absolute_difference(x14_face_lip_2,x14_face_lip_1)
        # loss4=tf.reduce_mean(tf.abs(x14_face_lip_1-x14_face_lip_2))
        # loss4: x17_face_lip_1, x17_face_lip_2
        loss5=tf.losses.absolute_difference(x17_face_lip_2,x17_face_lip_1)
        # loss5=tf.reduce_mean(tf.abs(x17_face_lip_1-x17_face_lip_2))
        # ?????
        loss=loss0+loss1+loss2+loss3+loss4+loss5
        return loss,loss0,loss1,loss2,loss3,loss4,loss5







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
            return tf.image.resize_images(bottom, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)




    def upsample_layer_m(self, bottom, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            nbatch = bottom.shape[0]
            old_height = bottom.shape[1]
            old_width = bottom.shape[2]
            upsampFactor = 2
            Ho = int(old_height)*upsampFactor
            Wo = int(old_width)*upsampFactor
            xi = tf.linspace(0., 1.0 * (int(old_height)), Ho)
            yi = tf.linspace(0., 1.0 * (int(old_width)), Wo)
            # sesstmp = tf.InteractiveSession()
            print int(old_height),int(old_width)
            yy, xx = tf.meshgrid(yi, xi)
            yy = tf.reshape(tf.transpose(yy), [yy.shape[0] * yy.shape[1], 1])
            xx = tf.reshape(tf.transpose(xx), [xx.shape[0] * xx.shape[1], 1])
            xxyy = tf.transpose(tf.concat([xx, yy], axis=1))
            g = xxyy
            g = g[:,:, tf.newaxis]
            g = tf.tile(g, [1, 1, nbatch])
            g = tf.reshape(g, [2, Ho, Wo, nbatch])
            g = tf.transpose(g, perm=[3, 1, 2, 0])
            output_upsample = tf.contrib.resampler.resampler(bottom, g)
            return output_upsample
            







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
                    return tf.layers.max_pooling2d(bottom,pool_size=ksize,strides=strides,padding='VALID',name=name)

    def relu_layer(self, bottom, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            bottom=tf.where(tf.is_nan(bottom),tf.ones_like(bottom)*0,bottom)
            return tf.nn.relu(bottom)

    def conv_layer(self, bottom, name):
        padding_mode=self.config_dict[name][2]
        if padding_mode=='valid':
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE) :
                if name in self.v201_param_list:
                    print '{} using pre-trained parameter'.format(name)
                    filt_key='{}.f'.format(name)
                    filt_init=tf.constant_initializer(self.v201_param_dict[filt_key])
                    filt=tf.get_variable('filter',self.config_dict[name][0],initializer=filt_init)
                    conv_biases_key='{}.b'.format(name)
                    conv_biases_init=tf.constant_initializer(self.v201_param_dict[conv_biases_key])
                    conv_biases=tf.get_variable('biases',self.config_dict[name][3],initializer=conv_biases_init)
                else:
                    print '{} not using pre-trained parameter'.format(name)
                    if self.USE_XAVIER==1:
                        filt = tf.get_variable('filter', self.config_dict[name][0], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                    elif self.USE_XAVIER==0:
                        filt=tf.get_variable('filter',self.config_dict[name][0],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32),dtype=tf.float32)
                    conv_biases=tf.get_variable('biases',self.config_dict[name][3],initializer=tf.constant_initializer(),dtype=tf.float32)
                strides=self.config_dict[name][1]
                conv = tf.nn.conv2d(bottom, filt, strides=strides, padding='VALID')
                bias = tf.nn.bias_add(conv, conv_biases)
                return bias
        elif isinstance(padding_mode,list):
            if len(padding_mode)==2:
                with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                    if name in self.v201_param_list:
                        print '{} using pre-trained parameter'.format(name)
                        filt_key='{}.f'.format(name)
                        filt_init=tf.constant_initializer(self.v201_param_dict[filt_key])
                        filt=tf.get_variable('filter',self.config_dict[name][0],initializer=filt_init)
                        conv_biases_key='{}.b'.format(name)
                        conv_biases_init=tf.constant_initializer(self.v201_param_dict[conv_biases_key])
                        conv_biases=tf.get_variable('biases',self.config_dict[name][3],initializer=conv_biases_init)
                    else:
                        print '{} not using pre-trained parameter'.format(name)
                        if self.USE_XAVIER==1:
                            filt = tf.get_variable('filter', self.config_dict[name][0], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                        elif self.USE_XAVIER==0:
                            filt=tf.get_variable('filter',self.config_dict[name][0],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32),dtype=tf.float32)
                        conv_biases=tf.get_variable('biases',self.config_dict[name][3],initializer=tf.constant_initializer(),dtype=tf.float32)
                    strides=self.config_dict[name][1]
                    paddings=tf.constant([[0, 0], [padding_mode[0], padding_mode[0]], [padding_mode[1], padding_mode[1]], [0, 0]])
                    bottom=tf.pad(bottom,paddings,"CONSTANT")
                    conv = tf.nn.conv2d(bottom, filt, strides=strides, padding='VALID')
                    bias = tf.nn.bias_add(conv, conv_biases)
                    return bias
            elif len(padding_mode)==4:
                with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
                    if name in self.v201_param_list:
                        print '{} using pre-trained parameter'.format(name)
                        filt_key='{}.f'.format(name)
                        filt_init=tf.constant_initializer(self.v201_param_dict[filt_key])
                        filt=tf.get_variable('filter',self.config_dict[name][0],initializer=filt_init)
                        conv_biases_key='{}.b'.format(name)
                        conv_biases_init=tf.constant_initializer(self.v201_param_dict[conv_biases_key])
                        conv_biases=tf.get_variable('biases',self.config_dict[name][3],initializer=conv_biases_init)
                    else:
                        print '{} not using pre-trained parameter'.format(name)
                        if self.USE_XAVIER==1:
                            filt = tf.get_variable('filter', self.config_dict[name][0], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                        elif self.USE_XAVIER==0:
                            filt=tf.get_variable('filter',self.config_dict[name][0],initializer=tf.truncated_normal_initializer(stddev=5e-2,dtype=tf.float32),dtype=tf.float32)
                        conv_biases=tf.get_variable('biases',self.config_dict[name][3],initializer=tf.constant_initializer(),dtype=tf.float32)
                    strides=self.config_dict[name][1]
                    paddings=tf.constant([[0, 0], [padding_mode[0], padding_mode[1]], [padding_mode[2], padding_mode[3]], [0, 0]])
                    bottom=tf.pad(bottom,paddings,"CONSTANT")
                    conv = tf.nn.conv2d(bottom, filt, strides=strides, padding='VALID')
                    bias = tf.nn.bias_add(conv, conv_biases)
                    return bias





    def bn_layer(self, bottom, name):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            #NHWC axis=batch(N)=0
            if name in self.v201_param_list:
                print '{} using pre-trained parameter'.format(name)
                gamma_key = '{}.m'.format(name)
                beta_key='{}.b'.format(name)
                mean_key='{}.x'.format(name)
                variance_key='{}.x'.format(name)
                gamma_init=tf.constant_initializer(self.v201_param_dict[gamma_key])
                beta_init=tf.constant_initializer(self.v201_param_dict[beta_key])
                mean_init=tf.constant_initializer(self.v201_param_dict[mean_key][:,0])
                variance_init=tf.constant_initializer(np.square(self.v201_param_dict[variance_key][:,1]))
                return tf.layers.batch_normalization(bottom,axis=-1,epsilon=1e-4,gamma_initializer=gamma_init,beta_initializer=beta_init,moving_mean_initializer=mean_init,moving_variance_initializer=variance_init)
            else:
                print '{} not using pre-trained parameter'.format(name)
                return tf.layers.batch_normalization(bottom,axis=-1,epsilon=1e-4)
    


    def read_v201_param(self, USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, AVOID_LAYERS_LIST, IDNUM):
        if IDNUM==1:
            # server009
            v201_param_path = '/media/server009/seagate/liuhan/lls/github/yousaidthat/v201_param_cell_idnum1.mat'
            # mac
            # v201_param_path = '/Users/lls/Documents/face/github/yousaidthat/v201_param_cell_idnum1.mat'
            # ssh
            # v201_param_path = '/workspace/liuhan/work/avasyn/github/yousaidthat/v201_param_cell_idnum1.mat'
            # ssh7
            # v201_param_path = '/media/data2/liuhan/github/yousaidthat/v201_param_cell_idnum1.mat'
        elif IDNUM == 5:
            # server009
            v201_param_path = '/media/server009/seagate/liuhan/lls/github/yousaidthat/v201_param_cell.mat'
            # mac
            # v201_param_path = '/Users/lls/Documents/face/github/yousaidthat/v201_param_cell.mat'
            # ssh
            # v201_param_path = '/workspace/liuhan/work/avasyn/github/yousaidthat/v201_param_cell.mat'
            # ssh7
            # v201_param_path = '/media/data2/liuhan/github/yousaidthat/v201_param_cell.mat'
        v201_param = scio.loadmat(v201_param_path)
        v201_param_dict = {}
        v201_param_list = []
        for i in range(0, v201_param['v201_param_cell'].shape[1]):
            key = str(v201_param['v201_param_cell'][0][i][0])
            value = v201_param['v201_param_cell'][1][i]
            v201_param_dict[key] = value
            if not key.split('.')[0] in v201_param_list:
                v201_param_list.append(key.split('.')[0])
        if USE_AUDIO == 0:
            for param_name in reversed(v201_param_list):
                if (param_name[0:2] == 'co' and param_name[-2:] == 'io'):
                    v201_param_list.remove(param_name)
                elif (param_name[0:2] == 'fc' and param_name[-2:] == 'io'):
                    v201_param_list.remove(param_name)
        if USE_FACE == 0:
            for param_name in reversed(v201_param_list):
                if (param_name[0:2] == 'co' and param_name[-2:] == 'ce'):
                    v201_param_list.remove(param_name)
        if USE_BN == 0:
            for param_name in reversed(v201_param_list):
                if param_name[0:2] == 'bn':
                    v201_param_list.remove(param_name)
        if USE_LIP == 0:
            for param_name in reversed(v201_param_list):
                if 'lip' in param_name:
                    v201_param_list.remove(param_name)
        if USE_DECODER == 0:
            for param_name in reversed(v201_param_list):
                if 'face' not in param_name and param_name[-2:]!='io':
                    v201_param_list.remove(param_name)
        if USE_FACEFC == 0 and USE_FACE == 1:
            v201_param_list.remove('conv6_face')
            v201_param_list.remove('conv7_face')
        if USE_AUDIOFC == 0 and USE_AUDIO == 1:
            v201_param_list.remove('fc6_audio')
            v201_param_list.remove('fc7_audio')
        if AVOID_LAYERS_LIST != None:
            for layer in AVOID_LAYERS_LIST:
                print '{} '.format(layer),
                v201_param_list.remove(layer)
            print 'WILL NOT BE INITIALIAZED'

            

        return v201_param_dict, v201_param_list



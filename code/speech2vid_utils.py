import numpy as np
import scipy.misc
import os


# three padding conditions
# 'VALID' zero: [0,0,0,0]
# const: [1,1,1,1] / [2,2,2,2] / ...
# diy: [1,0,1,0]

# conv: kernel, stride, padding, bias
# pool: kernel, stride, padding


def create_config_dict():
    config_dict = {
        'conv1_audio': [[3, 3, 1, 64], [1, 1, 1, 1], [1, 1], [64]],  # tf.pad
        'pool1_audio': [[1,1], [1, 1], 'valid'],
        'conv2_audio': [[3, 3, 64, 128], [1, 1, 1, 1], [1, 1], [128]],
        'pool2_audio': [[3, 3], [1, 2], [1, 0, 0, 0]],  # tf.concat
        'conv3_audio': [[3, 3, 128, 256], [1, 1, 1, 1], [1, 1], [256]],
        'conv4_audio': [[3, 3, 256, 256], [1, 1, 1, 1], [1, 1], [256]],
        'conv5_audio': [[3, 3, 256, 256], [1, 1, 1, 1], [1, 1], [256]],
        'pool5_audio': [[3, 3], [2, 2], 'valid'],
        'fc6_audio': [[5, 8, 256, 512], [1, 1, 1, 1], 'valid', [512]],
        'fc7_audio': [[1, 1, 512, 256], [1, 1, 1, 1], 'valid', [256]],

        'conv1_face': [[7, 7, 15, 96], [1, 2, 2, 1], 'valid', [96]],
        'conv1_face_idnum1': [[7, 7, 3, 96], [1, 2, 2, 1], 'valid', [96]],
        'pool1_face': [[3, 3], [2, 2], 'valid'],
        'conv2_face': [[5, 5, 96, 256], [1, 2, 2, 1], [1, 1], [256]],
        'pool2_face': [[3, 3], [2, 2], [0, 1, 0, 1]],
        'conv3_face': [[3, 3, 256, 512], [1, 1, 1, 1], [1, 1], [512]],
        'conv4_face': [[3, 3, 512, 512], [1, 1, 1, 1], [1, 1], [512]],
        'conv5_face': [[3, 3, 512, 512], [1, 1, 1, 1], [1, 1], [512]],
        'conv6_face': [[6, 6, 512, 512], [1, 1, 1, 1], 'valid', [512]],
        'conv7_face': [[1, 1, 512, 256], [1, 1, 1, 1], 'valid', [256]],

        'conv8': [[1, 1, 512, 128], [1, 1, 1, 1], 'valid', [128]],
        'conv1_1': [[3, 3, 128, 128], [1, 1, 1, 1], [1, 1], [128]],
        'conv1_2': [[3, 3, 128, 256], [1, 1, 1, 1], [1, 1], [256]],
        'conv1_3': [[3, 3, 256, 512], [1, 1, 1, 1], 'valid', [512]],
        'conv2': [[3, 3, 512, 256], [1, 1, 1, 1], [1, 1], [256]],
        'conv3_1': [[3, 3, 512, 256], [1, 1, 1, 1], [1, 1], [256]],
        'conv3_2': [[5, 5, 256, 96], [1, 1, 1, 1], [3, 3], [96]],
        'conv4': [[3, 3, 192, 96], [1, 1, 1, 1], [2, 1, 2, 1], [96]],
        'conv5_1': [[3, 3, 192, 64], [1, 1, 1, 1], [2, 1, 2, 1], [64]],
        'conv5_2': [[3, 3, 64, 32], [1, 1, 1, 1], [2, 1, 2, 1], [32]],
        'conv5_3': [[3, 3, 32, 3], [1, 1, 1, 1], [2, 1, 2, 1], [3]],

        'conv1_face_lip_1': [[7, 7, 3, 96], [1, 2, 2, 1], 'valid', [96]],
        # same as conv1_face_lip_1
        'conv1_face_lip_2': [[7, 7, 3, 96], [1, 2, 2, 1], 'valid', [96]],
        'pool1_face_lip_1': [[3, 3], [2, 2], 'valid'],
        # same as pool1_face_lip_1
        'pool1_face_lip_2': [[3, 3], [2, 2], 'valid'],
        'conv2_face_lip_1': [[5, 5, 96, 256], [1, 2, 2, 1], [1, 1], [256]],
        # same as conv2_face_lip_1
        'conv2_face_lip_2': [[5, 5, 96, 256], [1, 2, 2, 1], [1, 1], [256]],
        'pool2_face_lip_1': [[3, 3], [2, 2], [0, 1, 0, 1]],
        # same as pool2_face_lip_1
        'pool2_face_lip_2': [[3, 3], [2, 2], [0, 1, 0, 1]],
        'conv3_face_lip_1': [[3, 3, 256, 512], [1, 1, 1, 1], [1, 1], [512]],
        # same as conv3_face_lip_1
        'conv3_face_lip_2': [[3, 3, 256, 512], [1, 1, 1, 1], [1, 1], [512]],
        'conv4_face_lip_1': [[3, 3, 512, 512], [1, 1, 1, 1], [1, 1], [512]],
        # same as conv4_face_lip_1
        'conv4_face_lip_2': [[3, 3, 512, 512], [1, 1, 1, 1], [1, 1], [512]],
        'conv5_face_lip_1': [[3, 3, 512, 512], [1, 1, 1, 1], [1, 1], [512]],
        # same as conv5_face_lip_1
        'conv5_face_lip_2': [[3, 3, 512, 512], [1, 1, 1, 1], [1, 1], [512]],
        
        'upsamp1_1': {'filters': 128, 'kernel_size': 4, 'strides': 2, 'padding': [1,1]},
        'upsamp1_2': {'filters': 128, 'kernel_size': 4, 'strides': 2, 'padding': [1,1]},
        'upsamp1_3': {'filters': 256, 'kernel_size': 4, 'strides': 2, 'padding': [1,1]},
        'upsamp2': {'filters': 512, 'kernel_size': 4, 'strides': 2, 'padding': [1,1]},
        'upsamp3_1': {'filters': 512, 'kernel_size': 4, 'strides': 2, 'padding': [1,1]},
        'upsamp4': {'filters': 192, 'kernel_size': 4, 'strides': 2, 'padding': [1,1]},
        'upsamp5_1': {'filters': 192, 'kernel_size': 4, 'strides': 2, 'padding': [1,1]},
        

    }
    return config_dict



def network_variables():
    return {
        'audio':['conv1_audio','bn1_audio','relu1_audio','pool1_audio','conv2_audio','bn2_audio','relu2_audio','pool2_audio','conv3_audio','bn3_audio','relu3_audio','conv4_audio','bn4_audio','relu4_audio','conv5_audio','bn5_audio','relu5_audio','pool5_audio','fc6_audio','bn6_audio','relu6_audio','fc7_audio','bn7_audio','relu7_audio'],
        'identity':['conv1_face','bn1_face','relu1_face','pool1_face','conv2_face','bn2_face','relu2_face','pool2_face','conv3_face','bn3_face','relu3_face','conv4_face','bn4_face','relu4_face','conv5_face','bn5_face','relu5_face','conv6_face','bn6_face','relu6_face','conv7_face','bn7_face','relu7_face'],
        'decoder':['concat8','conv8','relu8','upsamp1_1','conv1_1','relu1_1','upsamp1_2','conv1_2','relu1_2','upsamp1_3','conv1_3','relu1_3','upsamp2','conv2','relu2','concatSkip2','upsamp3_1','conv3_1','relu3_1','conv3_2','relu3_2','concatSkip3','upsamp4','conv4','relu4','concatSkip4','upsamp5_1','conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','sigmoid5_3'],
        'lip':['conv1_face_lip_1', 'bn1_face_lip_1', 'relu1_face_lip_1', 'pool1_face_lip_1', 'conv2_face_lip_1','bn2_face_lip_1','relu2_face_lip_1','pool2_face_lip_1','conv3_face_lip_1','bn3_face_lip_1','relu3_face_lip_1','conv4_face_lip_1','bn4_face_lip_1','relu4_face_lip_1','conv5_face_lip_1','bn5_face_lip_1','relu5_face_lip_1']
    }






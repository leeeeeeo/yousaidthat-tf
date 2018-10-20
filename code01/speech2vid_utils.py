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
        'pool2_face_lip_1':[[3,3],[2,2],[0,1,0,1]],
        # same as pool2_face_lip_1
        'pool2_face_lip_2':[[3,3],[2,2],[0,1,0,1]],
        'conv3_face_lip_1':[[3,3,256,512],[1,1,1,1],[1,1],[512]],
        # same as conv3_face_lip_1
        'conv3_face_lip_2':[[3,3,256,512],[1,1,1,1],[1,1],[512]],
        'conv4_face_lip_1':[[3,3,512,512],[1,1,1,1],[1,1],[512]],
        # same as conv4_face_lip_1
        'conv4_face_lip_2':[[3,3,512,512],[1,1,1,1],[1,1],[512]],
        'conv5_face_lip_1':[[3,3,512,512],[1,1,1,1],[1,1],[512]],
        # same as conv5_face_lip_1
        'conv5_face_lip_2':[[3,3,512,512],[1,1,1,1],[1,1],[512]],
    }
    return config_dict



def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)



def load_image(image_path):
    face=imread(image_path)
    audio_path='{}-audio.png'.format(os.path.join(os.path.dirname(image_path),os.path.basename(image_path).split('-')[0]))
    audio=imread(audio_path)
    return face,audio

def load_face_image(image_path):
    face=imread(image_path)
    return face

def load_audio_image(image_path):
    audio_path='{}-audio.png'.format(os.path.join(os.path.dirname(image_path),os.path.basename(image_path).split('-')[0]))
    audio=imread(audio_path)
    return audio



def load_data(image_path,flip=True,is_test=False):
    face,audio=load_image(image_path)





# def mainUtils():
    # config_dict=create_config_dict()
    # for i in config_dict:
    #     if isinstance(config_dict[i][2],list):
    #         if len(config_dict[i][2])==4:
    #             print config_dict[i][2] 

if __name__=='__main__':
    mainUtils()
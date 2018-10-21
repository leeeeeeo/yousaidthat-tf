# import tensorflow as tf
# import numpy as np
# a = np.random.randint(0, 20, (12, 35*3))
# b = tf.convert_to_tensor(a, dtype=tf.float32)
# b = tf.reshape(b, [1, 12, 35, 3])
# print b.shape


# c = np.zeros((1, 0, 35, 3))
# print c.shape
# c = tf.convert_to_tensor(c, dtype=tf.float32)
# result = tf.concat([b, c], 1)
# print result.shape[0]


# d = np.zeros((2, 1, 0))
# print d


# import cv2
# img = cv2.imread('../../data/trump_12.png')
# SIFT = cv2.xfeatures2d_SIFT.create()
# from glob import glob
# data = glob('../headpose/code/exp01/h*')
# print (data)


# import numpy as np
# import cv2
# a = np.load(
#     '/Users/lls/Documents/face/data/lrw1018/lipread_mp4/MIGHT/test/MIGHT_00001_2.npz')
# face_gt = a['face_gt']
# mfcc_gt = a['mfcc_gt']
# identity1 = a['identity1']
# identity5 = a['identity5']
# print face_gt.shape
# print mfcc_gt.shape
# # cv2.imshow('a', face_gt)
# # cv2.waitKey(0)


from build_data_utils import read_and_decode
import tensorflow as tf
face_batch, audio_batch, identity5_batch = read_and_decode(
    '../../../data/lrw1018/lipread_mp4/MIGHT/test/test.tfrecords', 1, 5)
print face_batch.shape
print audio_batch.shape
print identity5_batch.shape
face_batch = tf.cast(face_batch, dtype=tf.float32)
audio_batch = tf.cast(audio_batch, dtype=tf.float32)
identity5_batch = tf.cast(identity5_batch, dtype=tf.float32)
print face_batch.shape
print audio_batch.shape
print identity5_batch.shape

# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
import speech2vid_train

try:
    from os import scandir
except ImportError:
    from scandir import scandir


TFRECORDS_PATH = '../../../data/lrw1016/lipread_mp4/MIGHT/test/test.tfrecords'
NPZ_PATH = '../../../data/lrw1016/lipread_mp4/MIGHT/test/'


def data_reader(input_dir, shuffle=True):
    file_paths = []
    for img_file in scandir(input_dir):
        if img_file.name.endswith('npz'):
            file_paths.append(img_file.path)
    if shuffle:
        shuffled_index = list(range(len(file_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)
        file_paths = [file_paths[i] for i in shuffled_index]
    return file_paths


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example(file_path):
    npz = np.load(file_path)
    face_gt = npz['face_gt'].tostring()
    mfcc_gt = npz['mfcc_gt'].tostring()
    identity1 = npz['identity1'].tostring()
    identity5 = npz['identity5'].tostring()
    file_name = file_path.split('/')[-1]
    example = tf.train.Example(features=tf.train.Features(feature={
        'file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
        'face_gt': _bytes_feature(face_gt),
        'mfcc_gt': _bytes_feature(mfcc_gt),
        'identity1': _bytes_feature(identity1),
        'identity5': _bytes_feature(identity5),
    }))
    return example


def data_writer(input_dir, output_file):
    file_paths = data_reader(input_dir)
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error as e:
        pass

    images_num = len(file_paths)
    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(len(file_paths)):
        file_path = file_paths[i]
        with tf.gfile.FastGFile(file_path, 'rb') as f:
            image_data = f.read()
        example = _convert_to_example(file_path)
        writer.write(example.SerializeToString())

        if i % 500 == 0:
            print 'processed {}/{}'.format(i, images_num)

    print 'done!'
    writer.close()


def read_and_decode(tfrecords_path, batch_size, identity_num):
    '''
    return: image[4D [batch size, width, height, channel]]
    '''
    if identity_num == 1:
        filename_queue = tf.train.string_input_producer([tfrecords_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'face_gt': tf.FixedLenFeature([], tf.string),
            'mfcc_gt': tf.FixedLenFeature([], tf.string),
            'identity1': tf.FixedLenFeature([], tf.string)
        })
        face_gt = tf.decode_raw(features['face_gt'], tf.uint8)
        face_gt = tf.reshape(face_gt, [109, 109, 3])
        face_gt = tf.image.per_image_standardization(face_gt)
        mfcc_gt = tf.decode_raw(features['mfcc_gt'], tf.float32)
        mfcc_gt = tf.reshape(mfcc_gt, [12, 35, 1])
        identity1 = tf.decode_raw(features['identity1'], tf.uint8)
        identity1 = tf.reshape(identity1, [112, 112, 3])
        identity1 = tf.image.per_image_standardization(identity1)
        face_gt_batch, mfcc_gt_batch, identity1_batch = tf.train.batch(
            [face_gt, mfcc_gt, identity1], batch_size=batch_size, num_threads=64, capacity=2000)
        return face_gt_batch, mfcc_gt_batch, identity1_batch
    elif identity_num == 5:
        filename_queue = tf.train.string_input_producer([tfrecords_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'face_gt': tf.FixedLenFeature([], tf.string),
            'mfcc_gt': tf.FixedLenFeature([], tf.string),
            'identity5': tf.FixedLenFeature([], tf.string)
        })
        face_gt = tf.decode_raw(features['face_gt'], tf.uint8)
        face_gt = tf.reshape(face_gt, [109, 109, 3])
        face_gt = tf.image.per_image_standardization(face_gt)
        mfcc_gt = tf.decode_raw(features['mfcc_gt'], tf.float32)
        mfcc_gt = tf.reshape(mfcc_gt, [12, 35, 1])
        identity5 = tf.decode_raw(features['identity5'], tf.uint8)
        identity5 = tf.reshape(identity5, [112, 112, 15])
        identity5 = tf.image.per_image_standardization(identity5)
        face_gt_batch, mfcc_gt_batch, identity5_batch = tf.train.batch(
            [face_gt, mfcc_gt,  identity5], batch_size=batch_size, num_threads=64, capacity=2000)
        return face_gt_batch, mfcc_gt_batch, identity5_batch


def main(argv=None):
    '''write tfrecords file'''
    tfrecord_output_file = TFRECORDS_PATH
    print 'convert face data to tfrecords'
    data_writer(NPZ_PATH, tfrecord_output_file)
    print 'finish face data to tfrecords'

    '''read tfrecords file'''
    # read_and_decode(TFRECORDS_PATH, BATCH_SIZE, 5)


if __name__ == '__main__':
    tf.app.run()

# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os
from datetime import datetime


def dlibShape2Array(landmark2D, dtype="int"):
    coords = []
    for i in range(0, 68):
        coords.append([int(landmark2D.part(i).x), int(landmark2D.part(i).y)])
    return np.asarray(coords)


def procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)), 0)
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection is not 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    if my < m:
        T = T[:my, :]
    c = muX - b*np.dot(muY, T)
    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform


def data_reader(input_dir, shuffle=True):
    file_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            img_file = os.path.join(root, file)
            if img_file.endswith('npz'):
                file_paths.append(img_file)
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


def _convert_to_example(file_path, passnan):
    npz = np.load(file_path)
    if passnan == 1:
        if not np.isnan(np.sum(npz['mfcc_gt'])):
            mfcc_gt = npz['mfcc_gt'].tostring()
            face_gt = npz['face_gt'].tostring()
            face_blur = npz['face_blur'].tostring()
            identity1 = npz['identity1'].tostring()
            identity5 = npz['identity5'].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'face_gt': _bytes_feature(face_gt),
                'mfcc_gt': _bytes_feature(mfcc_gt),
                'identity1': _bytes_feature(identity1),
                'identity5': _bytes_feature(identity5),
                'face_blur': _bytes_feature(face_blur),
            }))
            return example
        else:
            return - 1
    else:
        mfcc_gt = npz['mfcc_gt'].tostring()
        face_gt = npz['face_gt'].tostring()
        identity1 = npz['identity1'].tostring()
        identity5 = npz['identity5'].tostring()
        face_blur = npz['face_blur'].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'face_gt': _bytes_feature(face_gt),
            'mfcc_gt': _bytes_feature(mfcc_gt),
            'identity1': _bytes_feature(identity1),
            'identity5': _bytes_feature(identity5),
            'face_blur': _bytes_feature(face_blur),
        }))
        return example


def data_writer(input_dir, output_file, passnan):
    print 'READING NPZ'
    file_paths = data_reader(input_dir)
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error as e:
        pass

    images_num = len(file_paths)
    writer = tf.python_io.TFRecordWriter(output_file)
    print 'BUILDING TFRECORDS'

    for i in range(len(file_paths)):
        try:
            file_path = file_paths[i]
            with tf.gfile.FastGFile(file_path, 'rb') as f:
                image_data = f.read()
            example = _convert_to_example(file_path, passnan)
            if example == -1:
                print 'NAN: {}'.format(file_path)
                continue
            writer.write(example.SerializeToString())
        except:
            print 'FAIL: {}'.format(file_path)
            continue

        if i % 500 == 0:
            print '{} processing {} {}/{}'.format(
                datetime.now().strftime("%m/%d %H:%M:%S"), os.path.basename(output_file).split('.')[0], i, images_num)
    writer.close()


def parser_1(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
        'face_gt': tf.FixedLenFeature([], tf.string),
        'mfcc_gt': tf.FixedLenFeature([], tf.string),
        'identity1': tf.FixedLenFeature([], tf.string)
    })
    face_gt = tf.decode_raw(features['face_gt'], tf.uint8, name='face_0')
    face_gt = tf.reshape(face_gt, [109, 109, 3])
    mfcc_gt = tf.decode_raw(
        features['mfcc_gt'], tf.float64)
    mfcc_gt = tf.reshape(mfcc_gt, [12, 35, 1], name='audio_0')
    identity1 = tf.decode_raw(
        features['identity1'], tf.uint8)
    identity1 = tf.reshape(identity1, [112, 112, 3], name='identity_0')

    face_gt = tf.cast(face_gt, dtype=tf.float32)
    face_gt = tf.div(
        face_gt, tf.constant(255.0, dtype=tf.float32))
    mfcc_gt = tf.cast(mfcc_gt, dtype=tf.float32)
    identity1 = tf.cast(identity1, dtype=tf.float32)

    return face_gt, mfcc_gt, identity1


def parser_5(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
        'face_gt': tf.FixedLenFeature([], tf.string),
        'mfcc_gt': tf.FixedLenFeature([], tf.string),
        'identity5': tf.FixedLenFeature([], tf.string)
    })
    mfcc_gt = tf.decode_raw(
        features['mfcc_gt'], tf.float64)
    mfcc_gt = tf.reshape(mfcc_gt, [12, 35, 1], name='audio_0')
    identity5 = tf.decode_raw(
        features['identity5'], tf.uint8)
    identity5 = tf.reshape(identity5, [112, 112, 15], name='identity_0')
    face_gt = tf.decode_raw(features['face_gt'], tf.uint8)
    face_gt = tf.reshape(face_gt, [109, 109, 3], name='face_0')

    face_gt = tf.cast(face_gt, dtype=tf.float32)
    face_gt = tf.div(face_gt, tf.constant(255.0, dtype=tf.float32))
    mfcc_gt = tf.cast(mfcc_gt, dtype=tf.float32)
    identity5 = tf.cast(identity5, dtype=tf.float32)

    return face_gt, mfcc_gt, identity5


def parser_deblur(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
        'face_blur': tf.FixedLenFeature([], tf.string),
        'face_gt': tf.FixedLenFeature([], tf.string)
    })
    face_blur = tf.decode_raw(features['face_blur'], tf.uint8)
    face_blur = tf.reshape(face_blur, [109, 109, 3])
    face_blur = tf.cast(face_blur, dtype=tf.float32)
    face_blur = tf.div(face_blur, tf.constant(255.0, dtype=tf.float32))
    face_gt = tf.decode_raw(features['face_gt'], tf.uint8)
    face_gt = tf.reshape(face_gt, [109, 109, 3])
    face_gt = tf.cast(face_gt, dtype=tf.float32)
    face_gt = tf.div(face_gt, tf.constant(255.0, dtype=tf.float32))
    face_gt = tf.subtract(face_gt, face_blur)
    return face_blur, face_gt


def read_and_decode_TFRecordDataset(tfrecords_path, batch_size, identity_num, epoch_num):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    if identity_num == 1:
        dataset = dataset.map(parser_1).shuffle(buffer_size=100*batch_size)
    elif identity_num == 5:
        dataset = dataset.map(parser_5).shuffle(buffer_size=100*batch_size)
    epoch = tf.data.Dataset.range(epoch_num)
    dataset = epoch.flat_map(
        lambda i: tf.data.Dataset.zip(
            (dataset, tf.data.Dataset.from_tensors(i).repeat())))
    dataset = dataset.repeat(epoch_num).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    (face_gt_batch, mfcc_gt_batch, identity_batch), epoch_now = iterator.get_next()
    return face_gt_batch, mfcc_gt_batch, identity_batch, epoch_now


def read_and_decode_TFRecordDataset_deblur(tfrecords_path, batch_size, epoch_num):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parser_deblur).shuffle(buffer_size=100*batch_size)
    epoch = tf.data.Dataset.range(epoch_num)
    dataset = epoch.flat_map(lambda i: tf.data.Dataset.zip(
        (dataset, tf.data.Dataset.from_tensors(i).repeat())))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    (face_blur_batch, face_gt_batch), epochNow = iterator.get_next()
    return face_blur_batch, face_gt_batch, epochNow


def read_and_decode(tfrecords_path, batch_size, identity_num, epoch_num):
    '''
    return: image[4D [batch size, width, height, channel]]
    '''
    if identity_num == 1:
        filename_queue = tf.train.string_input_producer(
            tfrecords_path, num_epochs=epoch_num)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'face_gt': tf.FixedLenFeature([], tf.string),
            'mfcc_gt': tf.FixedLenFeature([], tf.string),
            'identity1': tf.FixedLenFeature([], tf.string)
        })
        face_gt = tf.decode_raw(features['face_gt'], tf.float64, name='face_0')
        face_gt = tf.reshape(face_gt, [109, 109, 3])
        mfcc_gt = tf.decode_raw(
            features['mfcc_gt'], tf.float64)
        mfcc_gt = tf.reshape(mfcc_gt, [12, 35, 1], name='audio_0')
        identity1 = tf.decode_raw(
            features['identity1'], tf.float64)
        identity1 = tf.reshape(identity1, [112, 112, 3], name='identity_0')
        '''tf.train.batch'''
        face_gt_batch, mfcc_gt_batch, identity1_batch = tf.train.batch(
            [face_gt, mfcc_gt, identity1], batch_size=batch_size, num_threads=64, capacity=2000)

        face_gt_batch = tf.cast(face_gt_batch, dtype=tf.float32)
        face_gt_batch = tf.div(
            face_gt_batch, tf.constant(255.0, dtype=tf.float32))
        mfcc_gt_batch = tf.cast(mfcc_gt_batch, dtype=tf.float32)
        identity1_batch = tf.cast(identity1_batch, dtype=tf.float32)

        return face_gt_batch, mfcc_gt_batch, identity1_batch
    elif identity_num == 5:
        filename_queue = tf.train.string_input_producer(tfrecords_path)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'face_gt': tf.FixedLenFeature([], tf.string),
            'mfcc_gt': tf.FixedLenFeature([], tf.string),
            'identity5': tf.FixedLenFeature([], tf.string)
        })
        mfcc_gt = tf.decode_raw(
            features['mfcc_gt'], tf.float64)
        mfcc_gt = tf.reshape(mfcc_gt, [12, 35, 1], name='audio_0')
        identity5 = tf.decode_raw(
            features['identity5'], tf.uint8)
        identity5 = tf.reshape(identity5, [112, 112, 15], name='identity_0')
        face_gt = tf.decode_raw(features['face_gt'], tf.uint8)
        face_gt = tf.reshape(face_gt, [109, 109, 3], name='face_0')

        '''tf.train.batch'''
        face_gt_batch, mfcc_gt_batch, identity5_batch = tf.train.batch(
            [face_gt, mfcc_gt,  identity5], batch_size=batch_size, num_threads=64, capacity=2000)
        '''tf.train.shuffle_batch'''
        # min_after_dequeue = 10000
        # capacity = min_after_dequeue + 3 * batch_size
        # face_gt_batch, mfcc_gt_batch, identity5_batch = tf.train.shuffle_batch([face_gt, mfcc_gt,  identity5],
        #                                                                        batch_size=batch_size,
        #                                                                        capacity=capacity,
        #                                                                        min_after_dequeue=min_after_dequeue)

        face_gt_batch = tf.cast(face_gt_batch, dtype=tf.float32)
        face_gt_batch = tf.div(
            face_gt_batch, tf.constant(255.0, dtype=tf.float32))
        mfcc_gt_batch = tf.cast(mfcc_gt_batch, dtype=tf.float32)
        identity5_batch = tf.cast(identity5_batch, dtype=tf.float32)

        return face_gt_batch, mfcc_gt_batch, identity5_batch


def create_test_npz_list(EARLY_STOPPING_FOLDER, EARLY_STOPPING_TEST_SIZE):
    '''read from txt file'''
    print '============ ERARLY STOPPING LIST EXISTED ============'
    if os.path.exists('../testNpzList.txt'):
        testNpzList = []
        txt = open('../testNpzList.txt', 'r')
        if EARLY_STOPPING_TEST_SIZE == 0:
            txtLines = txt.readlines()
        else:
            txtLines = txt.readlines()[:EARLY_STOPPING_TEST_SIZE]
        for txtLine in txtLines:
            txtLine = txtLine.strip('\n')
            testNpzList.append(txtLine)
        txt.close()
        print '============ TEST NPY LIST LENGTH {} ============'.format(
            len(testNpzList))
        return testNpzList

    print '============ READING ERARLY STOPPING TEST FOLDER ============'
    testNpzList = []
    for root, dirs, files in os.walk(EARLY_STOPPING_FOLDER):
        for file in files:
            if file.endswith('npz'):
                testNpzList.append(os.path.join(root, file))
    print '============ TEST NPY LIST LENGTH {} ============'.format(
        len(testNpzList))
    '''write to txt file'''
    txt = open('../testNpzList.txt', 'w')
    for testNpz in testNpzList:
        txt.write(testNpz + '\n')
    txt.close()
    return testNpzList

# -*- coding:utf-8 -*-


# 数据集文件夹：
# train/face_x.png
# train/audio_x.png

import os
import tensorflow as tf
import speech2vid_inference
from glob import glob
import numpy as np
import random
from speech2vid_utils import load_face_image, load_audio_image
import build_data


BATCH_SIZE = 8
EPOCH = 100000000
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 300000
TRAINING_SIZE = 100000000000
MODEL_SAVE_PATH = './models'
MODEL_NAME = 'test.ckpt'
LOG_SAVE_PATH = './logs'


def train():
    # input_audio = tf.placeholder(
    #     tf.float32, shape=[None, 12, 35, 1], name='input_audio')
    # input_face = tf.placeholder(
    #     tf.float32, shape=[None, 112, 112, 3], name='input_face')
    '''use tfrecords'''
    face_batch, audio_batch, identity5_batch = build_data.read_and_decode(
        build_data.TFRECORDS_PATH, BATCH_SIZE, 5)
    face_batch = tf.cast(face_batch, dtype=tf.float32)
    audio_batch = tf.cast(audio_batch, dtype=tf.float32)
    identity5_batch = tf.cast(identity5_batch, dtype=tf.float32)

    speech2vid = speech2vid_inference.Speech2Vid()
    prediction = speech2vid.inference(audio_batch, identity5_batch, BATCH_SIZE)
    train_loss = speech2vid.compute_loss(prediction, face_batch)

    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.GradientDescentOptimizer(
        LEARNING_RATE_BASE).minimize(train_loss, global_step=global_step)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(LOG_SAVE_PATH, sess.graph)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(TRAINING_STEPS):
            if coord.should_stop():
                break
            _, training_loss, step = sess.run(
                [train_op, train_loss, global_step])

            if step % 50 == 0:
                print 'Step {}, train loss = {}'.format(step, train_loss)
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step+1) == TRAINING_STEPS:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                              MODEL_NAME), global_step=global_step)
                print 'Model {} saved'.format(step)
    except tf.errors.OutOfRangeError:
        print 'Done training, epoch limit reached'
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

    '''use placeholder'''
    # placeholder
    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()

    #     faceList = glob('/path/to/train/folder/face-*.png')
    #     audioList = glob('/path/to/train/folder/audio-*.png')
    #     assert len(faceList) == len(audioList)
    #     shuffled_index = list(range(len(faceList)))
    #     random.seed(12345)
    #     random.shuffle(shuffled_index)
    #     faceList = [faceList[i] for i in shuffled_index]
    #     audioList = [audioList[i] for i in shuffled_index]
    #     batch_idxs = min(len(faceList), TRAINING_SIZE)//BATCH_SIZE

    #     for epoch in xrange(EPOCH):
    #         for idx in xrange(0, batch_idxs):
    #             face_batch_files = faceList[idx*BATCH_SIZE:(idx+1):BATCH_SIZE]
    #             face_batch = [load_face_image(face_batch_file)
    #                           for face_batch_file in face_batch_files]
    #             audio_batch_files = audioList[idx *
    #                                           BATCH_SIZE:(idx+1):BATCH_SIZE]
    #             audio_batch = [load_audio_image(
    #                 audio_batch_file) for audio_batch_file in audio_batch_files]

    #             _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={
    #                 input_face: face_batch, input_audio: audio_batch})

    #             print 'Epoch {} {}, loss {}'.format(epoch, idx, loss_value)

    #             if step % 1000 == 0:
    #                 saver.save(sess, os.path.join(MODEL_SAVE_PATH,
    #                                               MODEL_NAME), global_step=global_step)
    #                 print 'Model {} saved'.format(step)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()

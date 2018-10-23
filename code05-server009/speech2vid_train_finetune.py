# -*- coding:utf-8 -*-


# 数据集文件夹：
# train/face_x.png
# train/audio_x.png

import os
import tensorflow as tf
import speech2vid_inference_finetune
from glob import glob
import numpy as np
import random
from speech2vid_utils import load_face_image, load_audio_image
from build_data_utils import read_and_decode
import argparse
import logging
import traceback
import time


BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.05
LEARNING_RATE_DECAY = 0.96
TRAINING_STEPS = 300000
MODEL_SAVE_PATH = './models'
MODEL_NAME = 'test.ckpt'
LOG_SAVE_PATH = './logs'
NUM_GPUS = 2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


parser = argparse.ArgumentParser()
parser.add_argument(
    '--tfrecords', help='tfrecords path', default='../../../data/lrw1021/lipread_mp4/MIGHT/test/test.tfrecords', type=str)
args = parser.parse_args()


def train():
    '''use tfrecords'''
    face_batch, audio_batch, identity5_batch = read_and_decode(
        args.tfrecords, BATCH_SIZE, 5)
    face_batch = tf.cast(face_batch, dtype=tf.float32)
    audio_batch = tf.cast(audio_batch, dtype=tf.float32)
    identity5_batch = tf.cast(identity5_batch, dtype=tf.float32)

    # for i in xrange(NUM_GPUS):
    #     with tf.device('/gpu:%d' % i):

    speech2vid = speech2vid_inference_finetune.Speech2Vid()
    prediction = speech2vid.inference(
        audio_batch, identity5_batch, BATCH_SIZE)
    train_loss = speech2vid.compute_loss(
        prediction, face_batch)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, 5000, LEARNING_RATE_DECAY, staircase=False)
    train_op = tf.train.GradientDescentOptimizer(
        LEARNING_RATE_BASE).minimize(train_loss, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(LOG_SAVE_PATH, graph)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            start_time = time.time()
            for step in np.arange(TRAINING_STEPS):
                if coord.should_stop():
                    break

                _,  training_loss, step = sess.run(
                    [train_op,  train_loss, global_step])

                if step % 500 == 0:
                    end_time = time.time()
                    elapsed_time = end_time-start_time
                    print 'Step: {}, Train loss: {},  Elapsed time: {}'.format(
                        step,  training_loss, elapsed_time)
                    summary_str = sess.run(summary_op)
                    # train_writer.add_summary(summary_str, step)
                    start_time = time.time()

                if step % 10000 == 0 or (step+1) == TRAINING_STEPS:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                                  MODEL_NAME), global_step=global_step)
                    print 'Model {} saved'.format(step)
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            traceback.print_exc()
            coord.request_stop(e)
        finally:
            saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                          MODEL_NAME), global_step=global_step)
            print 'Model {} saved'.format(step)
            coord.request_stop()
            coord.join(threads)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()

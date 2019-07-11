# -*- coding:utf-8 -*-

import argparse
import collections
import os
import time
import traceback
from datetime import datetime

import tensorflow as tf

import deblur_inference
from build_data_utils import read_and_decode_TFRecordDataset_deblur, create_test_npz_list
from deblur_early_stopping import deblur_early_stopping

parser = argparse.ArgumentParser()
parser.add_argument('--tfrecords', help='tfrecords path', nargs='+')
parser.add_argument('--gpu', help='which gpu?', type=str)
parser.add_argument('--loss_func', help='mse / l2', type=str)
parser.add_argument('--use_pretrained_conv',
                    help='use pretrained conv? 1 / 0.', type=int)
parser.add_argument('--name', help='model name', type=str)
parser.add_argument('--baselr', default=0,
                    help='default=0.00001', type=float)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--epochnum', default=20, type=int)
parser.add_argument('--max_to_keep', type=int)
args = parser.parse_args()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.name == 'time':
    name = time.strftime('%H%M%S', time.localtime())
else:
    name = '{}-{}'.format(args.name, time.strftime('%H%M%S', time.localtime()))

MAX_TO_KEEP = args.max_to_keep
LOSS_FUNCTION = args.loss_func
USE_PRETRAINED_CONV = args.use_pretrained_conv
BASIC_LEARNING_RATE = tf.constant(args.baselr, dtype=tf.float32)
BATCH_SIZE = args.batchsize
EPOCH_NUM = args.epochnum
TRAINING_STEPS = 3000000
MODEL_SAVE_PATH = '../../models/{}/{}'.format(os.path.abspath(os.path.join(
    os.path.dirname("__file__"), os.path.pardir)).split('/')[-1], name)
MODEL_NAME = '{}.ckpt'.format(name)
Model = collections.namedtuple(
    "Model", "epochNow,inputGt,inputBlur,outputDeblur,deblurLoss,train")


def create_model():
    faceBlurBatch, faceGtBatch, epochNow = read_and_decode_TFRecordDataset_deblur(
        args.tfrecords, BATCH_SIZE, EPOCH_NUM)
    deblur = deblur_inference.Deblur(USE_PRETRAINED_CONV)
    with tf.variable_scope('deblur_inference'):
        deblurPrediction = deblur.inference(faceBlurBatch)
    with tf.variable_scope('deblur_loss'):
        deblurLoss = deblur.compute_loss(
            deblurPrediction, faceGtBatch, LOSS_FUNCTION)
    with tf.variable_scope('deblur_train'):
        deblurOptimizer = tf.train.AdamOptimizer(
            learning_rate=BASIC_LEARNING_RATE)
        deblurGradsAndVars = deblurOptimizer.compute_gradients(deblurLoss)
        deblurTrain = deblurOptimizer.apply_gradients(deblurGradsAndVars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    updateLoss = ema.apply([deblurLoss])
    globalStep = tf.train.get_or_create_global_step()
    increaseGlobalStep = tf.assign(globalStep, globalStep + 1)

    return Model(
        epochNow=epochNow,
        inputGt=faceGtBatch,
        inputBlur=faceBlurBatch,
        outputDeblur=deblurPrediction,
        deblurLoss=deblurLoss,
        train=tf.group(updateLoss, increaseGlobalStep, deblurTrain)
    )


def train():
    model = create_model()

    with tf.name_scope("parameter_count"):
        parameterCount = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
    sv = tf.train.Supervisor(logdir=MODEL_SAVE_PATH,
                             save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        print '============ TOTAL PARAMETER COUNT: {} ============'.format(
            sess.run(parameterCount))

        try:
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            saver.save(sess, os.path.join(
                MODEL_SAVE_PATH, MODEL_NAME), global_step=0)
            print("============ MODEL {} SAVED ============".format(0))

        except KeyboardInterrupt:
            print '============ KEYBOARD INTERRPUTED ============'
        except Exception as e:
            traceback.print_exc()


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()

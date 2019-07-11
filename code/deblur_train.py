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
parser.add_argument('--early_stopping_folder',
                    help='test folder used for early stopping', type=str)
parser.add_argument('--gpu', help='which gpu?', type=str)
parser.add_argument('--loss_func', help='mse / l2', type=str)
parser.add_argument('--use_pretrained_conv',
                    help='use pretrained conv? 1 / 0.', type=int)
parser.add_argument('--name', help='model name', type=str)
parser.add_argument('--ckpt', help='checkpoint path to restore', type=str)
parser.add_argument('--baselr', default=0.000001,
                    help='default=0.00001', type=float)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--epochnum', default=20, type=int)
parser.add_argument("--summary_freq", type=int, default=50,
                    help="update summaries every summary_freq steps")
parser.add_argument("--display_freq", type=int, default=100,
                    help="display progress every display_freq steps")
parser.add_argument("--trace_freq", type=int, default=0,
                    help="trace execution every trace_freq steps")
parser.add_argument("--save_freq", type=int, default=1000,
                    help="save model every save_freq steps, 0 to disable")
parser.add_argument("--early_stopping_freq", type=int, default=5000,
                    help="test every early_stopping_freq steps")
parser.add_argument('--saveinanotherdir',
                    help='continue from ckpt and save in another dir?', type=int)
parser.add_argument("--early_stopping_size", type=int, default=2000)
parser.add_argument('--max_to_keep', type=int)
args = parser.parse_args()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.name == 'time':
    name = time.strftime('%H%M%S', time.localtime())
else:
    name = '{}-{}'.format(args.name, time.strftime('%H%M%S', time.localtime()))
if args.ckpt:
    if args.saveinanotherdir == 1:
        name = '{}-{}'.format(args.name,
                              time.strftime('%H%M%S', time.localtime()))
    else:
        if args.ckpt.split('/')[-1] == '':
            name = args.ckpt.split('/')[-2]
        else:
            name = args.ckpt.split('/')[-1]

MAX_TO_KEEP = args.max_to_keep
EARLY_STOPPING_TEST_SIZE = args.early_stopping_size
LOSS_FUNCTION = args.loss_func
EARLY_STOPPING_FOLDER = args.early_stopping_folder
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
    earlyStoppingLossList = []
    testNpzList = create_test_npz_list(
        EARLY_STOPPING_FOLDER, EARLY_STOPPING_TEST_SIZE)
    model = create_model()
    tf.summary.scalar("deblurLoss", model.deblurLoss)
    tf.summary.image("inputBlur", model.inputBlur)
    tf.summary.image("inputGt", model.inputGt)
    tf.summary.image("outputDeblur", model.outputDeblur)

    with tf.name_scope("parameter_count"):
        parameterCount = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
    sv = tf.train.Supervisor(logdir=MODEL_SAVE_PATH,
                             save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        print '============ TOTAL PARAMETER COUNT: {} ============'.format(
            sess.run(parameterCount))
        if args.ckpt:
            ckpt = tf.train.get_checkpoint_state(args.ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                ckptFile = open('{}/checkpoint'.format(args.ckpt), 'r')
                STEP_START = int(ckptFile.readlines()[
                                 0].split("\"")[1].split('-')[-1])
                print '============ CHECKPOINT {} LOADED ============'.format(
                    ckpt.model_checkpoint_path)

            '''load early stopping log'''
            if os.path.exists('{}/{}'.format(args.ckpt, 'early_stopping.txt')):
                earlyStoppingTxt = open(
                    '{}/{}'.format(args.ckpt, 'early_stopping.txt'), 'r')
                earlyStoppingLines = earlyStoppingTxt.readlines()
                for earlyStoppingLine in earlyStoppingLines:
                    earlyStoppingStep, earlyStoppingLoss = earlyStoppingLine.strip(
                        '\n').split(',')
                    earlyStoppingLossList.append(
                        (int(earlyStoppingStep), float(earlyStoppingLoss)))
                earlyStoppingTxt.close()
            else:
                earlyStoppingLossList = []
            if args.saveinanotherdir == 1:
                earlyStoppingLossList = []
            print 'earlyStoppingLossList', earlyStoppingLossList
        else:
            STEP_START = 0
            if os.path.exists('{}/{}'.format(MODEL_SAVE_PATH, 'early_stopping.txt')):
                os.remove('{}/{}'.format(MODEL_SAVE_PATH, 'early_stopping.txt'))

        try:
            start = time.time()
            for step in range(STEP_START, TRAINING_STEPS):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == TRAINING_STEPS - 1)
                options = None
                run_metadata = None
                if should(args.trace_freq):
                    options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    'train': model.train,
                    'globalStep': sv.global_step,
                }

                if should(args.display_freq):
                    fetches['deblurLoss'] = model.deblurLoss
                    fetches['epochNow'] = model.epochNow

                if should(args.summary_freq):
                    fetches['summary'] = sv.summary_op

                results = sess.run(fetches, options=options,
                                   run_metadata=run_metadata)

                if should(args.summary_freq):
                    sv.summary_writer.add_summary(
                        results["summary"], results["globalStep"])

                if should(args.display_freq):
                    print '{} {} EPOCH: {} STEP: {}\tDEBLUR_LOSS: {}\tTIME: {}'.format(datetime.now().strftime(
                        "%m/%d %H:%M:%S"), name, results["epochNow"][0], results['globalStep'], results["deblurLoss"], time.time() - start)
                    start = time.time()
                    if results["epochNow"][0] > EPOCH_NUM:
                        print '============ REACH MAX EPOCH ============'
                        return 0

                if should(args.save_freq):
                    if not os.path.exists(MODEL_SAVE_PATH):
                        os.makedirs(MODEL_SAVE_PATH)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                                  MODEL_NAME), global_step=sv.global_step)
                    print("============ MODEL {} SAVED ============".format(
                        results['globalStep']))

                if should(args.early_stopping_freq):
                    earlyStoppingLoss = deblur_early_stopping(
                        testNpzList, MODEL_SAVE_PATH, USE_PRETRAINED_CONV, earlyStoppingLossList, LOSS_FUNCTION)
                    earlyStoppingLossList.append(
                        (results['globalStep'], earlyStoppingLoss))
                    '''write early stopping log'''
                    earlyStoppingTxt = open(
                        '{}/{}'.format(MODEL_SAVE_PATH, 'early_stopping.txt'), 'a')
                    earlyStoppingTxt.write(
                        '{},{}\n'.format(results['globalStep'], earlyStoppingLoss))
                    earlyStoppingTxt.close()

                    if len(earlyStoppingLossList) > 10:
                        print 'earlyStoppingLossList', earlyStoppingLossList[len(
                            earlyStoppingLossList) - 10:]
                        lastTenLossList = [earlyStoppingLoss[1] for earlyStoppingLoss in earlyStoppingLossList[len(
                            earlyStoppingLossList) - 10:]]
                        if min(lastTenLossList) == lastTenLossList[0]:
                            print '============ EARLY STOPPING AT {} ============'.format(
                                earlyStoppingLossList[len(earlyStoppingLossList)-10][0])
                            return 0

        except KeyboardInterrupt:
            print '============ KEYBOARD INTERRPUTED ============'
        except Exception as e:
            traceback.print_exc()
        finally:
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                          MODEL_NAME), global_step=sv.global_step)
            print("============ MODEL {} SAVED ============".format(
                results['globalStep']))


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()

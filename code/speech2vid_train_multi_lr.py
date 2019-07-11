# -*- coding:utf-8 -*-


import os
import tensorflow as tf

import numpy as np
from build_data_utils import read_and_decode_TFRecordDataset
import argparse
import logging
import traceback
import time
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument(
    '--tfrecords', help='tfrecords path', nargs='+')
parser.add_argument('--gpu', help='which gpu?', default='1', type=str)
parser.add_argument('--bn', help='use bn pre-trained params?', type=int)
parser.add_argument(
    '--audio', help='use audio conv pre-trained params?', type=int)
parser.add_argument(
    '--audiofc', help='use audio fc pre-trained params?', type=int)
parser.add_argument(
    '--face', help='use identity conv pre-trained params?', type=int)
parser.add_argument(
    '--facefc', help='use identity fc pre-trained params?', type=int)
parser.add_argument(
    '--decoder', help='use decoder pre-trained params?', type=int)
parser.add_argument(
    '--lip', help='use loss encoder pre-trained params?', type=int)
parser.add_argument(
    '--avoidlayers', help='layers do not need initialization', nargs='+')
parser.add_argument('--momentum', help='momentum', type=float)
parser.add_argument('--name', help='model name', type=str)
parser.add_argument('--ckpt', help='checkpoint path to restore', type=str)
parser.add_argument('--baselr', default=0.00001,
                    help='default=0.00001', type=float)
parser.add_argument('--audiolr', default=0.05, help='default=0.05', type=float)
parser.add_argument('--identitylr', default=1, help='default=1', type=float)
parser.add_argument('--bnlr', default=1, help='default=1', type=float)
parser.add_argument('--liplr', default=0, help='default=0', type=float)
parser.add_argument('--decoderlr', default=1, help='default=1', type=float)
parser.add_argument('--idnum', help='identity number: 1 or 5?', type=int)
parser.add_argument('--saveinanotherdir',
                    help='continue from ckpt and save in another dir?', type=int)
parser.add_argument('--batchsize', default=50, type=int)
parser.add_argument('--epochnum', default=20, type=int)
parser.add_argument('--earlystopstep', default=20, type=int)
parser.add_argument('--mode', help="deconv / upsample", type=str)
parser.add_argument(
    '--xavier', help='use xavier_initializer or truncated_normal_initializer', type=int)
args = parser.parse_args()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.name == 'time':
    name = time.strftime('%H%M%S', time.localtime())
else:
    name = '{}-{}_idnum{}'.format(args.name, time.strftime('%H%M%S',
                                                           time.localtime()), args.idnum)
if args.ckpt:
    if args.saveinanotherdir == 1:
        name = '{}-{}_idnum{}'.format(args.name, time.strftime('%H%M%S',
                                                               time.localtime()), args.idnum)
    else:
        name = args.ckpt.split('/')[-1]

if args.mode == 'deconv':
    import speech2vid_inference_deconv
elif args.mode == 'upsample':
    import speech2vid_inference_finetune

BATCH_SIZE = args.batchsize
EPOCH_NUM = args.epochnum
BASIC_LEARNING_RATE = tf.constant(args.baselr, dtype=tf.float32)
IDENTITY_LEARNING_RATE_BASE = tf.constant(args.identitylr, dtype=tf.float32)
AUDIO_LEARNING_RATE_BASE = tf.constant(args.audiolr, dtype=tf.float32)
BN_LEARNING_RATE_BASE = tf.constant(args.bnlr, dtype=tf.float32)
LIP_LEARNING_RATE_BASE = tf.constant(args.liplr, dtype=tf.float32)
DECODER_LEARNING_RATE_BASE = tf.constant(args.decoderlr, dtype=tf.float32)
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 300000
MODEL_SAVE_PATH = '../models/{}'.format(name)
MODEL_NAME = '{}.ckpt'.format(name)
LOG_SAVE_PATH = '../logs/{}'.format(name)
NUM_GPUS = 2
USE_BN = args.bn
USE_AUDIO = args.audio
USE_LIP = args.lip
USE_DECODER = args.decoder
USE_FACE = args.face
USE_FACEFC = args.facefc
USE_AUDIOFC = args.audiofc
USE_XAVIER = args.xavier
MOMENTUM = args.momentum
AVOID_LAYERS_LIST = args.avoidlayers
EARLY_STOP_STEP = args.earlystopstep


def train():
    '''use tfrecords'''
    face_batch, audio_batch, identity5_batch, epoch_now = read_and_decode_TFRecordDataset(
        args.tfrecords, BATCH_SIZE, args.idnum, EPOCH_NUM)

    speech2vid = speech2vid_inference_finetune.Speech2Vid(
        USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, args.idnum)
    audio_encoder_output = speech2vid.audio_encoder(audio_batch)
    identity_encoder_output, x7_face, x4_face, x3_face = speech2vid.identity_encoder(
        identity5_batch, args.idnum)
    prediction = speech2vid.image_decoder(
        audio_encoder_output, identity_encoder_output, x7_face, x4_face, x3_face)
    # prediction = speech2vid.inference(audio_batch, identity5_batch)
    train_loss, loss0, loss1, loss2, loss3, loss4, loss5 = speech2vid.loss_l1(
        prediction, face_batch)

    bn_var_list = [var for var in tf.trainable_variables()
                   if 'bn' in var.op.name]
    audio_var_list = [var for var in tf.trainable_variables()
                      if 'audio' in var.op.name and var not in bn_var_list]
    identity_var_list = [
        var for var in tf.trainable_variables() if str(var.op.name).split('/')[0][-4:] == 'face' and var not in bn_var_list]
    lip_var_list = [
        var for var in tf.trainable_variables() if 'face_lip' in var.op.name and var not in bn_var_list]
    decoder_var_list = [var for var in tf.trainable_variables(
    ) if var not in audio_var_list + identity_var_list + lip_var_list + bn_var_list]

    global_step = tf.Variable(0, trainable=False)
    '''exponential_decay lr'''
    # identity_learning_rate = tf.train.exponential_decay(
    #     IDENTITY_LEARNING_RATE_BASE, global_step, 5000, LEARNING_RATE_DECAY, staircase=False)
    # audio_learning_rate = tf.train.exponential_decay(
    #     AUDIO_LEARNING_RATE_BASE, global_step, 5000, LEARNING_RATE_DECAY, staircase=False)
    # bn_learning_rate = tf.train.exponential_decay(
    #     BN_LEARNING_RATE_BASE, global_step, 5000, LEARNING_RATE_DECAY, staircase=False)
    # lip_learning_rate = tf.train.exponential_decay(
    #     LIP_LEARNING_RATE_BASE, global_step, 5000, LEARNING_RATE_DECAY, staircase=False)
    # decoder_learning_rate = tf.train.exponential_decay(
    #     DECODER_LEARNING_RATE_BASE, global_step, 5000, LEARNING_RATE_DECAY, staircase=False)
    '''constant lr'''
    identity_learning_rate = BASIC_LEARNING_RATE*IDENTITY_LEARNING_RATE_BASE
    audio_learning_rate = BASIC_LEARNING_RATE*AUDIO_LEARNING_RATE_BASE
    bn_learning_rate = BASIC_LEARNING_RATE*BN_LEARNING_RATE_BASE
    lip_learning_rate = BASIC_LEARNING_RATE*LIP_LEARNING_RATE_BASE
    decoder_learning_rate = BASIC_LEARNING_RATE*DECODER_LEARNING_RATE_BASE
    '''SGD'''
    # identity_optimizer = tf.train.GradientDescentOptimizer(identity_learning_rate)
    # audio_optimizer = tf.train.GradientDescentOptimizer(audio_learning_rate)
    # bn_optimizer = tf.train.GradientDescentOptimizer(bn_learning_rate)
    # lip_optimizer = tf.train.GradientDescentOptimizer(lip_learning_rate)
    # decoder_optimizer = tf.train.GradientDescentOptimizer(decoder_learning_rate)
    '''Momentum'''
    # identity_optimizer = tf.train.MomentumOptimizer(
    #     identity_learning_rate, MOMENTUM)
    # audio_optimizer = tf.train.MomentumOptimizer(audio_learning_rate, MOMENTUM)
    # bn_optimizer = tf.train.MomentumOptimizer(bn_learning_rate, MOMENTUM)
    # lip_optimizer = tf.train.MomentumOptimizer(lip_learning_rate, MOMENTUM)
    # decoder_optimizer = tf.train.MomentumOptimizer(
    #     decoder_learning_rate, MOMENTUM)
    '''Adam'''
    identity_optimizer = tf.train.AdamOptimizer(
        learning_rate=identity_learning_rate)
    audio_optimizer = tf.train.AdamOptimizer(learning_rate=audio_learning_rate)
    bn_optimizer = tf.train.AdamOptimizer(learning_rate=bn_learning_rate)
    lip_optimizer = tf.train.AdamOptimizer(learning_rate=lip_learning_rate)
    decoder_optimizer = tf.train.AdamOptimizer(
        learning_rate=decoder_learning_rate)
    '''Seperate learning rate option 1'''
    identity_train_op = identity_optimizer.minimize(
        train_loss, global_step=global_step, var_list=identity_var_list)
    bn_train_op = bn_optimizer.minimize(
        train_loss, global_step=global_step, var_list=bn_var_list)
    lip_train_op = lip_optimizer.minimize(
        train_loss, global_step=global_step, var_list=lip_var_list)
    decoder_train_op = decoder_optimizer.minimize(
        train_loss, global_step=global_step, var_list=decoder_var_list)
    audio_train_op = audio_optimizer.minimize(
        train_loss, global_step=global_step, var_list=audio_var_list)
    train_op = tf.group(identity_train_op, audio_train_op,
                        bn_train_op, lip_train_op, decoder_train_op)
    '''Only train decoder'''
    # decoder_train_op = decoder_optimizer.minimize(
    #     train_loss, global_step=global_step, var_list=decoder_var_list)
    # train_op = tf.group(decoder_train_op)
    '''Seperate learning rate option 2'''
    # grads = tf.gradients(train_loss, bn_var_list+audio_var_list +
    #                      identity_var_list+lip_var_list+decoder_var_list)
    # bn_grad = grads[:len(bn_var_list)]
    # audio_grad = grads[len(bn_var_list):len(bn_var_list + audio_var_list)]
    # identity_grad = grads[len(bn_var_list + audio_var_list)                          :len(bn_var_list + audio_var_list + identity_var_list)]
    # lip_grad = grads[len(bn_var_list + audio_var_list + identity_var_list)                     :len(bn_var_list + audio_var_list + identity_var_list + lip_var_list)]
    # decoder_grad = grads[len(
    #     bn_var_list + audio_var_list + identity_var_list + lip_var_list):]
    # identity_train_op = identity_optimizer.apply_gradients(
    #     zip(identity_grad, identity_var_list), global_step=global_step)
    # bn_train_op = bn_optimizer.apply_gradients(
    #     zip(bn_grad, bn_var_list), global_step=global_step)
    # lip_train_op = lip_optimizer.apply_gradients(
    #     zip(lip_grad, lip_var_list), global_step=global_step)
    # decoder_train_op = decoder_optimizer.apply_gradients(
    #     zip(decoder_grad, decoder_var_list), global_step=global_step)
    # audio_train_op = audio_optimizer.apply_gradients(
    #     zip(audio_grad, audio_var_list), global_step=global_step)
    # train_op = tf.group(identity_train_op, audio_train_op,
    #                     bn_train_op, lip_train_op, decoder_train_op)
    '''Only one learning rate'''
    # optimizer = tf.train.GradientDescentOptimizer(identity_learning_rate)
    # train_op = optimizer.minimize(train_loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=0)

    tf.summary.scalar("loss", train_loss)
    tf.summary.image("face_gt",  face_batch)
    tf.summary.image("audio",  audio_batch)
    tf.summary.image("prediction",  prediction)

    summary_op = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if args.ckpt:
            ckpt = tf.train.get_checkpoint_state(args.ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print '{} loaded'.format(ckpt.model_checkpoint_path)

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_writer = tf.summary.FileWriter(LOG_SAVE_PATH, sess.graph)
        early_stop_loss_list = []
        try:
            start_time = time.time()
            for step in np.arange(TRAINING_STEPS):
                # if coord.should_stop():
                #     break

                # _,  training_loss, step, summary, identity_learning_rate_, audio_learning_rate_, bn_learning_rate_, lip_learning_rate_, decoder_learning_rate_, loss0_, loss1_, loss2_, loss3_, loss4_, loss5_, audio_encoder_output_, identity_encoder_output_, x7_face_, base_lr_ = sess.run(
                #     [train_op,  train_loss, global_step, summary_op, identity_optimizer._learning_rate, audio_optimizer._learning_rate, bn_optimizer._learning_rate, lip_optimizer._learning_rate, decoder_optimizer._learning_rate, loss0, loss1, loss2, loss3, loss4, loss5, audio_encoder_output, identity_encoder_output, x7_face, BASIC_LEARNING_RATE])

                '''When using Adam'''
                _,  training_loss, step, summary, loss0_, loss1_, loss2_, loss3_, loss4_, loss5_, audio_encoder_output_, identity_encoder_output_, x7_face_, base_lr_, epoch_now_ = sess.run(
                    [train_op,  train_loss, global_step, summary_op, loss0, loss1, loss2, loss3, loss4, loss5, audio_encoder_output, identity_encoder_output, x7_face, BASIC_LEARNING_RATE, epoch_now])

                train_writer.add_summary(summary, step)
                # print 'x7_face_', np.mean(x7_face_)
                # print 'audio_encoder_output', np.max(audio_encoder_output_), np.min(
                #     audio_encoder_output_), np.mean(audio_encoder_output_)
                # print 'identity_encoder_output', np.max(identity_encoder_output_), np.min(
                #     identity_encoder_output_), np.mean(identity_encoder_output_)

                if step % 1 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    # print '{}  Step: {}  Total loss: {}\tLoss0: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tLoss5: {}\tTime: {}\tBASE Lr: {}  ID Lr: {}  AU Lr: {}  BN Lr: {}  LIP Lr: {}  DE Lr: {}'.format(
                    #     name, step,  round(training_loss, 5), round(loss0_, 5), round(loss1_, 5), round(loss2_, 5), round(loss3_, 5), round(loss4_, 5), round(loss5_, 5), round(elapsed_time, 2), round(base_lr_, 10), round(identity_learning_rate_, 10), round(audio_learning_rate_, 10), round(bn_learning_rate_, 10), round(lip_learning_rate_, 10), round(decoder_learning_rate_, 10))

                    '''When using Adam'''
                    print '{}  Adam  {}  Epoch: {}  Step: {}  Total loss: {}\tLoss0: {}\tLoss1: {}\tLoss2: {}\tLoss3: {}\tLoss4: {}\tLoss5: {}\tTime: {}\tBASE Lr: {}'.format(datetime.now().strftime("%m/%d %H:%M:%S"),
                                                                                                                                                                              name, epoch_now_[0], step,  round(training_loss, 5), round(loss0_, 5), round(loss1_, 5), round(loss2_, 5), round(loss3_, 5), round(loss4_, 5), round(loss5_, 5), round(elapsed_time, 2), round(base_lr_, 10))

                    start_time = time.time()

                if step % 1000 == 0 or (step+1) == TRAINING_STEPS:
                    if not os.path.exists(MODEL_SAVE_PATH):
                        os.makedirs(MODEL_SAVE_PATH)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                                  MODEL_NAME), global_step=global_step)
                    print 'Model {} saved at {}'.format(
                        step, MODEL_SAVE_PATH)

                    '''Early stopping'''
                    # 1. test on validation set, record loss and the minimum loss
                    # 2. if loss on validation set is larger than the minimum loss for 10 times, stop training

        except KeyboardInterrupt:
            logging.info('Interrupted')
            # coord.request_stop()
        except Exception as e:
            traceback.print_exc()
            # coord.request_stop(e)
        finally:
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                          MODEL_NAME), global_step=global_step)
            print 'Model {} saved at {}'.format(step, MODEL_SAVE_PATH)
            # coord.request_stop()
            # coord.join(threads)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()

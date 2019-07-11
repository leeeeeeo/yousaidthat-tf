# -*- coding:utf-8 -*-


import os
import tensorflow as tf
import speech2vid_inference_finetune
import numpy as np
from build_data_utils import read_and_decode
from speech2vid_utils import network_variables
import argparse
import logging
import traceback
import time


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
parser.add_argument(
    '--xavier', help='use xavier_initializer or truncated_normal_initializer', type=int)
args = parser.parse_args()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
name = args.name
if name == 'time':
    name = time.strftime('%m%d-%H%M%S', time.localtime())
else:
    name = '{}-{}'.format(name, time.strftime('%m%d-%H%M%S', time.localtime()))
if args.ckpt:
    name = args.ckpt.split('/')[-1]

BATCH_SIZE = 32
BASIC_LEARNING_RATE = tf.constant(args.baselr, dtype=tf.float32)
IDENTITY_LEARNING_RATE_BASE = tf.constant(args.identitylr, dtype=tf.float32)
AUDIO_LEARNING_RATE_BASE = tf.constant(args.audiolr, dtype=tf.float32)
BN_LEARNING_RATE_BASE = tf.constant(args.bnlr, dtype=tf.float32)
LIP_LEARNING_RATE_BASE = tf.constant(args.liplr, dtype=tf.float32)
DECODER_LEARNING_RATE_BASE = tf.constant(args.decoderlr, dtype=tf.float32)
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 300000
MODEL_SAVE_PATH = './models/{}'.format(name)
MODEL_NAME = '{}.ckpt'.format(name)
LOG_SAVE_PATH = './logs/{}'.format(name)
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


def train():
    '''use tfrecords'''
    face_batch, audio_batch, identity5_batch = read_and_decode(
        args.tfrecords, BATCH_SIZE, args.idnum)
    face_batch = tf.cast(face_batch, dtype=tf.float32)
    audio_batch = tf.cast(audio_batch, dtype=tf.float32)
    identity5_batch = tf.cast(identity5_batch, dtype=tf.float32)

    speech2vid = speech2vid_inference_finetune.Speech2Vid(
        USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST)
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
    ) if var not in audio_var_list+identity_var_list+lip_var_list+bn_var_list]

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
    identity_optimizer = tf.train.MomentumOptimizer(
        identity_learning_rate, MOMENTUM)
    audio_optimizer = tf.train.MomentumOptimizer(audio_learning_rate, MOMENTUM)
    bn_optimizer = tf.train.MomentumOptimizer(bn_learning_rate, MOMENTUM)
    lip_optimizer = tf.train.MomentumOptimizer(lip_learning_rate, MOMENTUM)
    decoder_optimizer = tf.train.MomentumOptimizer(
        decoder_learning_rate, MOMENTUM)
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
    '''Seperate learning rate option 2'''
    # grads = tf.gradients(train_loss, identity_var_list+audio_var_list)
    # identity_grad = grads[:len(identity_var_list)]
    # audio_grad = grads[len(identity_var_list):]
    # identity_train_op = identity_optimizer.apply_gradients(
    #     zip(identity_grad, identity_var_list))
    # audio_train_op = audio_optimizer.apply_gradients(
    #     zip(audio_grad, audio_var_list))
    # train_op = tf.group(identity_train_op, audio_train_op)
    '''Only one learning rate'''
    # optimizer = tf.train.GradientDescentOptimizer(identity_learning_rate)
    # train_op = optimizer.minimize(train_loss, global_step=global_step)

    saver = tf.train.Saver()

    tf.summary.scalar("loss", train_loss)
    tf.summary.image("prediction",  prediction)
    tf.summary.image("face_gt",  face_batch)
    tf.summary.image("audio",  audio_batch)

    summary_op = tf.summary.merge_all()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if args.ckpt:
            ckpt = tf.train.get_checkpoint_state(args.ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print '{} loaded'.format(ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_writer = tf.summary.FileWriter(LOG_SAVE_PATH, sess.graph)
        try:
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                          MODEL_NAME), global_step=global_step)
            print 'Model {} saved at {}'.format(0, MODEL_SAVE_PATH)

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            traceback.print_exc()
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()

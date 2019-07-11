# -*- coding:utf-8 -*-


import os
import tensorflow as tf
import collections
import numpy as np
from build_data_utils import read_and_decode_TFRecordDataset, create_test_npz_list
import argparse
import logging
import traceback
import time
from datetime import datetime
import math
import speech2vid_early_stopping_gan_halfface


parser = argparse.ArgumentParser()
parser.add_argument(
    '--tfrecords', help='tfrecords path', nargs='+')
parser.add_argument('--early_stopping_folder',
                    help='test folder used for early stopping', type=str)
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
parser.add_argument('--l1weight', type=float)
parser.add_argument('--ganweight', type=float)
parser.add_argument('--earlystopstep', default=20, type=int)
parser.add_argument('--mode', help="deconv / upsample / gan", type=str)
parser.add_argument(
    '--xavier', help='use xavier_initializer or truncated_normal_initializer', type=int)
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
parser.add_argument("--early_stopping_size", type=int, default=2000)
parser.add_argument('--max_to_keep', type=int)
parser.add_argument('--is_training', type=int)
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
        if args.ckpt.split('/')[-1] == '':
            name = args.ckpt.split('/')[-2]
        else:
            name = args.ckpt.split('/')[-1]

IS_TRAINING = args.is_training
MAX_TO_KEEP = args.max_to_keep
EPS = 1e-12
EARLY_STOPPING_TEST_SIZE = args.early_stopping_size
EARLY_STOPPING_FOLDER = args.early_stopping_folder
GAN_WEIGHT = args.ganweight*1.0
L1_WEIGHT = args.l1weight*1.0
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
MODEL_SAVE_PATH = '../../models/{}/{}'.format(os.path.abspath(os.path.join(
    os.path.dirname("__file__"), os.path.pardir)).split('/')[-1], name)
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
Model = collections.namedtuple(
    "Model", "epoch_now,input_groundtruth,input_audio,prediction, discriminator_real, discriminator_pred, discrim_loss, discrim_grads_and_vars,identity_grads_and_vars,audio_grads_and_vars,bn_grads_and_vars,lip_grads_and_vars,decoder_grads_and_vars, gen_loss_gan, gen_loss_l1, train")


def create_model():
    groundtruth_batch, audio_batch, identity_batch, epoch_now = read_and_decode_TFRecordDataset(
        args.tfrecords, BATCH_SIZE, args.idnum, EPOCH_NUM)

    if args.mode == 'deconv':
        import speech2vid_inference_deconv
        speech2vid = speech2vid_inference_deconv.Speech2Vid(
            USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, args.idnum)
    elif args.mode == 'upsample':
        import speech2vid_inference_finetune
        speech2vid = speech2vid_inference_finetune.Speech2Vid(
            USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, args.idnum)
    elif args.mode == 'gan_halfface':
        import speech2vid_inference_gan_halfface
        speech2vid = speech2vid_inference_gan_halfface.Speech2Vid(
            USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, args.idnum, IS_TRAINING)

    with tf.variable_scope('generator'):
        print '============ BUILD G ============'
        audio_encoder_output = speech2vid.audio_encoder(audio_batch)
        if args.idnum == 1:
            tf.summary.image("identity1", identity_batch)
        elif args.idnum == 5:
            tf.summary.image("identity1", tf.split(identity_batch, 5, -1)[0])
            tf.summary.image("identity2", tf.split(identity_batch, 5, -1)[1])
            tf.summary.image("identity3", tf.split(identity_batch, 5, -1)[2])
            tf.summary.image("identity4", tf.split(identity_batch, 5, -1)[3])
            tf.summary.image("identity5", tf.split(identity_batch, 5, -1)[4])
        identity_encoder_output, x7_face, x4_face, x3_face = speech2vid.identity_encoder(
            identity_batch, args.idnum)
        prediction = speech2vid.image_decoder(
            audio_encoder_output, identity_encoder_output, x7_face, x4_face, x3_face)

    with tf.name_scope('groundtruth_discriminator'):
        with tf.variable_scope('discriminator'):
            discriminator_real = speech2vid.discriminator(groundtruth_batch)

    with tf.name_scope('prediction_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            discriminator_pred = speech2vid.discriminator(prediction)

    with tf.name_scope('discriminator_loss'):
        discrim_loss = tf.reduce_mean(
            - (tf.log(discriminator_real + EPS) + tf.log(1 - discriminator_pred + EPS)))

    with tf.name_scope("generator_loss"):
        gen_loss_gan = tf.reduce_mean(-tf.log(discriminator_pred + EPS))
        gen_loss_l1, loss0, loss1, loss2, loss3, loss4, loss5 = speech2vid.loss_l1(
            prediction, groundtruth_batch)
        gen_loss = gen_loss_gan * GAN_WEIGHT + gen_loss_l1 * L1_WEIGHT

    with tf.name_scope('discriminator_train'):
        print '============ BUILD D ============'
        discrim_tvars = [var for var in tf.trainable_variables(
        ) if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(0.001)
        discrim_grads_and_vars = discrim_optim.compute_gradients(
            discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_var_list = [
                var for var in tf.trainable_variables() if 'generator' in var.op.name]
            bn_var_list = [var for var in tf.trainable_variables()
                           if 'bn' in var.op.name]
            audio_var_list = [var for var in tf.trainable_variables()
                              if 'audio' in var.op.name and var not in bn_var_list]
            identity_var_list = [
                var for var in tf.trainable_variables() if str(var.op.name).split('/')[1][-4:] == 'face' and var not in bn_var_list]
            lip_var_list = [
                var for var in tf.trainable_variables() if 'face_lip' in var.op.name and var not in bn_var_list]
            decoder_var_list = list(set(gen_var_list)-set(bn_var_list) -
                                    set(audio_var_list) - set(identity_var_list) - set(lip_var_list))
            identity_learning_rate = BASIC_LEARNING_RATE * IDENTITY_LEARNING_RATE_BASE
            identity_optimizer = tf.train.AdamOptimizer(
                learning_rate=identity_learning_rate)
            identity_grads_and_vars = identity_optimizer.compute_gradients(
                gen_loss, var_list=identity_var_list)
            identity_train_op = identity_optimizer.apply_gradients(
                identity_grads_and_vars)
            audio_learning_rate = BASIC_LEARNING_RATE*AUDIO_LEARNING_RATE_BASE
            audio_optimizer = tf.train.AdamOptimizer(
                learning_rate=audio_learning_rate)
            audio_grads_and_vars = audio_optimizer.compute_gradients(
                gen_loss, var_list=audio_var_list)
            audio_train_op = audio_optimizer.apply_gradients(
                audio_grads_and_vars)
            bn_learning_rate = BASIC_LEARNING_RATE*BN_LEARNING_RATE_BASE
            bn_optimizer = tf.train.AdamOptimizer(
                learning_rate=bn_learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                bn_grads_and_vars = bn_optimizer.compute_gradients(
                    gen_loss, var_list=bn_var_list)
                bn_train_op = bn_optimizer.apply_gradients(bn_grads_and_vars)
            lip_learning_rate = BASIC_LEARNING_RATE*LIP_LEARNING_RATE_BASE
            lip_optimizer = tf.train.AdamOptimizer(
                learning_rate=lip_learning_rate)
            lip_grads_and_vars = lip_optimizer.compute_gradients(
                gen_loss, var_list=lip_var_list)
            lip_train_op = lip_optimizer.apply_gradients(lip_grads_and_vars)
            decoder_learning_rate = BASIC_LEARNING_RATE*DECODER_LEARNING_RATE_BASE
            decoder_optimizer = tf.train.AdamOptimizer(
                learning_rate=decoder_learning_rate)
            decoder_grads_and_vars = decoder_optimizer.compute_gradients(
                gen_loss, var_list=decoder_var_list)
            decoder_train_op = decoder_optimizer.apply_gradients(
                decoder_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_gan, gen_loss_l1])
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        epoch_now=epoch_now,
        input_audio=audio_batch,
        input_groundtruth=groundtruth_batch,
        prediction=prediction,
        discriminator_real=discriminator_real,
        discriminator_pred=discriminator_pred,

        discrim_grads_and_vars=discrim_grads_and_vars,
        identity_grads_and_vars=identity_grads_and_vars,
        audio_grads_and_vars=audio_grads_and_vars,
        bn_grads_and_vars=bn_grads_and_vars,
        lip_grads_and_vars=lip_grads_and_vars,
        decoder_grads_and_vars=decoder_grads_and_vars,

        discrim_loss=ema.average(discrim_loss),
        gen_loss_gan=ema.average(gen_loss_gan),
        gen_loss_l1=ema.average(gen_loss_l1),

        train=tf.group(update_losses, incr_global_step, identity_train_op,
                       audio_train_op, bn_train_op, lip_train_op, decoder_train_op)

    )


def train():
    earlyStoppingLossList = []
    testNpzList = create_test_npz_list(
        EARLY_STOPPING_FOLDER, EARLY_STOPPING_TEST_SIZE)
    model = create_model()
    tf.summary.scalar("discrim_loss", model.discrim_loss)
    tf.summary.scalar("gen_loss_gan", model.gen_loss_gan)
    tf.summary.scalar("gen_loss_l1", model.gen_loss_l1)

    tf.summary.image("input_groundtruth",  model.input_groundtruth)
    tf.summary.image("input_audio",  model.input_audio)
    tf.summary.image("prediction", model.prediction)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.identity_grads_and_vars + model.audio_grads_and_vars + model.bn_grads_and_vars + model.lip_grads_and_vars + model.decoder_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)

    sv = tf.train.Supervisor(logdir=MODEL_SAVE_PATH,
                             save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        print '============ TOTAL PARAMETER COUNT: {} ============'.format(
            sess.run(parameter_count))
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
                    'global_step': sv.global_step,
                }

                if should(args.display_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_gan"] = model.gen_loss_gan
                    fetches["gen_loss_l1"] = model.gen_loss_l1
                    fetches["epoch_now"] = model.epoch_now

                if should(args.summary_freq):
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches, options=options,
                                   run_metadata=run_metadata)

                if should(args.summary_freq):
                    sv.summary_writer.add_summary(
                        results["summary"], results["global_step"])

                if should(args.trace_freq):
                    print("============ RECORDING TRACE ============")
                    sv.summary_writer.add_run_metadata(
                        run_metadata, "step_%d" % results["global_step"])

                if should(args.display_freq):
                    print '{} {} EPOCH: {} STEP: {}\tD_LOSS: {}\tG_GAN_LOSS: {}\tG_L1_LOSS: {}\tTIME: {}'.format(datetime.now().strftime(
                        "%m/%d %H:%M:%S"), name, results["epoch_now"][0], results['global_step'], results["discrim_loss"], results["gen_loss_gan"], results["gen_loss_l1"], time.time() - start)
                    start = time.time()
                    if results["epoch_now"][0] > EPOCH_NUM:
                        print '============ REACH MAX EPOCH ============'
                        return 0

                if should(args.save_freq):
                    if not os.path.exists(MODEL_SAVE_PATH):
                        os.makedirs(MODEL_SAVE_PATH)
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                                  MODEL_NAME), global_step=sv.global_step)
                    print("============ MODEL {} SAVED ============".format(
                        results['global_step']))

                if should(args.early_stopping_freq):
                    earlyStoppingLoss = speech2vid_early_stopping_gan_halfface.speech2vid_early_stopping_gan_halfface(testNpzList, MODEL_SAVE_PATH, earlyStoppingLossList, USE_AUDIO, USE_BN,
                                                                                                                      USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, args.idnum, GAN_WEIGHT, L1_WEIGHT, IS_TRAINING=0)
                    earlyStoppingLossList.append(
                        (results['global_step'], earlyStoppingLoss))
                    '''write early stopping log'''
                    earlyStoppingTxt = open(
                        '{}/{}'.format(MODEL_SAVE_PATH, 'early_stopping.txt'), 'a')
                    earlyStoppingTxt.write(
                        '{},{}\n'.format(results['global_step'], earlyStoppingLoss))
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
            print ('============ KEYBOARD INTERRPUTED ============')
        except Exception as e:
            traceback.print_exc()
        finally:
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
            saver.save(sess, os.path.join(MODEL_SAVE_PATH,
                                          MODEL_NAME), global_step=sv.global_step)
            print("============ MODEL {} SAVED ============".format(
                results['global_step']))


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()

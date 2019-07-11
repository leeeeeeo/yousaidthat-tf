import tensorflow as tf
import speech2vid_inference_gan_halfface
import numpy as np
from tqdm import tqdm

EPS = 1e-12


def norm1(imageNp):
    return imageNp / 255.0


def create_evaluation_model(mfccPlaceholder, identityPlaceholder, groundtruthPlaceholder, USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, ID_NUM, GAN_WEIGHT, L1_WEIGHT, IS_TRAINING):
    speech2vid = speech2vid_inference_gan_halfface.Speech2Vid(
        USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, ID_NUM, IS_TRAINING)
    with tf.variable_scope('generator'):
        audioEncoder = speech2vid.audio_encoder(mfccPlaceholder)
        identityEncoder, x7_face, x4_face, x3_face = speech2vid.identity_encoder(
            identityPlaceholder, ID_NUM)
        prediction = speech2vid.image_decoder(
            audioEncoder, identityEncoder, x7_face, x4_face, x3_face)
    with tf.name_scope('prediction_discriminator'):
        with tf.variable_scope('discriminator'):
            discriminatorPrediction = speech2vid.discriminator(prediction)
    with tf.name_scope("generator_loss"):
        generatorLossGan = tf.reduce_mean(
            -tf.log(discriminatorPrediction + EPS))
        generatorLossL1, _, _, _, _, _, _ = speech2vid.loss_l1(
            prediction, groundtruthPlaceholder)
        generatorLoss = generatorLossGan * GAN_WEIGHT + generatorLossL1 * L1_WEIGHT
    return generatorLoss


def speech2vid_early_stopping_gan_halfface(testNpzList, CHECKPOINT_DIR, earlyStoppingLossList, USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, ID_NUM, GAN_WEIGHT, L1_WEIGHT, IS_TRAINING):
    testLossList = []
    testGraph = tf.Graph()
    with testGraph.as_default():
        mfccPlaceholder = tf.placeholder(tf.float32, shape=[1, 12, 35, 1])
        identityPlaceholder = tf.placeholder(
            tf.float32, shape=[1, 112, 112, 3])
        groundtruthPlaceholder = tf.placeholder(
            tf.float32, shape=[1, 109, 109, 3])
        generatorLoss = create_evaluation_model(mfccPlaceholder, identityPlaceholder, groundtruthPlaceholder, USE_AUDIO, USE_BN,
                                                USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, ID_NUM, GAN_WEIGHT, L1_WEIGHT, IS_TRAINING)
        saver = tf.train.Saver()

    with tf.Session(graph=testGraph) as sess:
        ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if ckpt:
            saver.restore(sess, ckpt)
            ckptFile = open('{}/checkpoint'.format(CHECKPOINT_DIR), 'r')
            ckptStep = ckptFile.readlines()[0].split("\"")[1].split('-')[-1]
            print '============ LOAD CHECKPOINT {} SUCCESS ============'.format(
                ckptStep)
        else:
            print '============ LOAD CHECKPOINT FAIL ============'
        for testNpzPath in tqdm(testNpzList, desc='{} TH EARLY STOPPING'.format(len(earlyStoppingLossList) + 1)):
            testNpz = np.load(testNpzPath)
            identity = np.reshape(testNpz['identity1'].astype(
                np.float32), (1, 112, 112, 3))
            mfcc = np.reshape(testNpz['mfcc_gt'].astype(
                np.float32), (1, 12, 35, 1))
            groundtruth = np.reshape(
                norm1(testNpz['face_gt'].astype(np.float32)), (1, 109, 109, 3))
            generatorLoss_ = sess.run([generatorLoss], feed_dict={
                mfccPlaceholder: mfcc,
                identityPlaceholder: identity,
                groundtruthPlaceholder: groundtruth})
            testLossList.append(generatorLoss_[0])

    assert len(testLossList) == len(testNpzList)
    return 1.0*sum(testLossList)/len(testLossList)

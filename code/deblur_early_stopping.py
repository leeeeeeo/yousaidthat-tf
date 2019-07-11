import tensorflow as tf
import numpy as np
import deblur_inference
from tqdm import tqdm


def norm1(imageNp):
    return imageNp / 255.0


def create_evaluation_model(inputImagePlaceholder, USE_PRETRAINED_CONV):
    deblur = deblur_inference.Deblur(USE_PRETRAINED_CONV)
    with tf.variable_scope('deblur_inference'):
        outputResidual = deblur.inference(inputImagePlaceholder)
    return outputResidual


def deblur_early_stopping(testNpzList, CHECKPOINT_DIR, USE_PRETRAINED_CONV, earlyStoppingLossList, LOSS_FUNCTION):
    testLossList = []
    testGraph = tf.Graph()
    with testGraph.as_default():
        inputImagePlaceholder = tf.placeholder(
            tf.float32, shape=[1, 109, 109, 3])
        outputResidual = create_evaluation_model(
            inputImagePlaceholder, USE_PRETRAINED_CONV)

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
        for testNpzPath in tqdm(testNpzList, desc='{} TH EARLY STOPPING'.format(len(earlyStoppingLossList)+1)):
            testNpz = np.load(testNpzPath)
            faceBlur = np.reshape(
                norm1(testNpz['face_blur']), (1, 109, 109, 3))
            faceGt = norm1(testNpz['face_gt'])
            outputResidual_ = sess.run([outputResidual], feed_dict={
                inputImagePlaceholder: faceBlur})
            outputResidual_ = np.squeeze(outputResidual_)
            faceBlur = np.squeeze(faceBlur)
            outputImage = faceBlur + outputResidual_
            if LOSS_FUNCTION == 'mse':
                loss = sess.run(tf.losses.mean_squared_error(tf.convert_to_tensor(
                    faceGt), tf.convert_to_tensor(outputImage)))
            elif LOSS_FUNCTION == 'l2':
                loss = sess.run(tf.nn.l2_loss(tf.convert_to_tensor(
                    outputImage)-tf.convert_to_tensor(faceGt)))
            testLossList.append(loss)

    assert len(testLossList) == len(testNpzList)
    return 1.0*sum(testLossList)/len(testLossList)

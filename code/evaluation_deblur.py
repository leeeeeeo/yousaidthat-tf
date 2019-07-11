import tensorflow as tf
import argparse
import os
import numpy as np
from PIL import Image
import deblur_inference
np.set_printoptions(threshold='nan')

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--gpu', type=str)
parser.add_argument('--input', help='input image path', type=str)
parser.add_argument('--use_pretrained_conv',
                    help='use pretrained conv? 1 / 0.', type=int)
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

CHECKPOINT_DIR = args.ckpt
INPUT_IMAGE_PATH = args.input
USE_PRETRAINED_CONV = args.use_pretrained_conv
OUTPUT_FOLDER = '../outputs'


def norm255(imageNp):
    return imageNp * 255.0


def norm1(imageNp):
    return imageNp / 255.0


def PILshow(imageNp):
    if len(imageNp.shape) > 3:
        imageNp = np.squeeze(imageNp)
    imageNp = norm255(imageNp)
    image = Image.fromarray(imageNp.astype(np.uint8))
    image.show()
    return image


def create_evaluation_model(inputImagePlaceholder):
    deblur = deblur_inference.Deblur(USE_PRETRAINED_CONV)
    with tf.variable_scope('deblur_inference'):
        outputResidual = deblur.inference(inputImagePlaceholder)
    return outputResidual


def evaluation():
    inputImage = Image.open(INPUT_IMAGE_PATH)
    inputImage = inputImage.resize((109, 109))
    inputImage = np.array(inputImage)
    inputImage = np.reshape(inputImage, (1, 109, 109, 3))
    if np.max(inputImage) > 1:
        inputImage = norm1(inputImage)

    PILshow(inputImage)
    inputImagePlaceholder = tf.placeholder(tf.float32, shape=[1, 109, 109, 3])
    outputResidual = create_evaluation_model(inputImagePlaceholder)

    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=CHECKPOINT_DIR,
                             save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if ckpt:
            saver.restore(sess, ckpt)
            ckptFile = open('{}/checkpoint'.format(CHECKPOINT_DIR), 'r')
            latestCkptPath = ckptFile.readlines()[0].split("\"")[1]
            ckptStep, ckptName = latestCkptPath.split(
                '-')[-1], latestCkptPath.split('.')[0]
            print '============ LOAD CHECKPOINT {} SUCCESS ============'.format(
                ckptStep)
        else:
            print '============ LOAD CHECKPOINT FAIL ============'

        outputResidual_ = sess.run([outputResidual], feed_dict={
            inputImagePlaceholder: inputImage})
        outputImage = inputImage + outputResidual_[0]
        PILshow(outputResidual_[0])
        outputImage = PILshow(outputImage)

        OUTPUT_IMAGE_PATH = '{output_folder}/{codexx}/{model_name}/{input_image_name}-{step}.bmp'.format(
            output_folder=OUTPUT_FOLDER,
            codexx=os.path.abspath(os.path.join(os.path.dirname(
                "__file__"), os.path.pardir)).split('/')[-1],
            model_name=ckptName,
            input_image_name=os.path.splitext(
                os.path.split(INPUT_IMAGE_PATH)[1])[0],
            step=ckptStep)
        if not os.path.exists(os.path.split(OUTPUT_IMAGE_PATH)[0]):
            os.makedirs(os.path.split(OUTPUT_IMAGE_PATH)[0])
        outputImage.save(OUTPUT_IMAGE_PATH)
        print '============ DEBLUR SAVED ============'
        print '============ DEBLUR DONE ============'


if __name__ == "__main__":
    evaluation()

import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import os


def create_config_dict():
    config_dict = {
        'supr0': [[3, 3, 3, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr1': [[3, 3, 64, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr2': [[3, 3, 64, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr3': [[3, 3, 64, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr4': [[3, 3, 64, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr5': [[3, 3, 64, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr6': [[3, 3, 64, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr7': [[3, 3, 64, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr8': [[3, 3, 64, 64], [1, 1, 1, 1], [1, 1, 1, 1], [64]],
        'supr9': [[3, 3, 64, 3], [1, 1, 1, 1], [1, 1, 1, 1], [3]],
    }
    return config_dict


def build_blur_images():
    INPUT_IMAGE_PATH = '/media/server009/seagate/liuhan/lls/github/yousaidthat-tf/images/faceimg1/faceimg1_1.png'
    OUTPUT_IMAGE_PATH = '{}/{}_blur.png'.format(os.path.dirname(
        INPUT_IMAGE_PATH), os.path.basename(INPUT_IMAGE_PATH).split('.')[0])
    inputImage = Image.open(INPUT_IMAGE_PATH)
    inputImageNp = np.array(inputImage)
    blurImageNp = cv2.GaussianBlur(inputImageNp, (5, 5), 2)
    blurImage = Image.fromarray(blurImageNp)
    blurImage.save(OUTPUT_IMAGE_PATH)


if __name__ == "__main__":
    build_blur_images()

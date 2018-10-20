import cv2
import dlib
import os
import random
import scipy.io as scio
import glob
from scipy.io import wavfile
import subprocess
import numpy as np
import python_speech_features
import traceback
import time
import argparse
from build_data_utils import data_writer, procrustes, dlibShape2Array

'''
NOTICE: lrw1016, lrw1017, lrw1018...
'''

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', help='which dataset to process', default='lrw', type=str)
parser.add_argument('--device', help='which device?', type=str)
parser.add_argument(
    '--func', help='txt: generate dataset mp4 path txt, data: preprocess dataset， tfrecords: build tfrecords', type=str)
args = parser.parse_args()


def baseface(img):
    dets = dlibDetector(img, 1)
    V = dlibPredictor(img, dets[0])
    V = dlibShape2Array(V)[27:48]
    d, Z, transform = procrustes(U, V, scaling=True, reflection=False)
    T = np.array([[0, 0, 0], [0, 0, 0], [transform['translation']
                                         [0], transform['translation'][1], 1]])
    T[0:2, 0:2] = transform['rotation'] * transform['scale']
    T = T[:, 0:2].T
    dst = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]))
    dst = dst[0:240, 0:240, :]
    dst = dst[22:22+197, 22:22+197, :]
    dst = cv2.resize(dst, (112, 112))
    return dst


def extract_face_list(frame_list):
    '''
    function: extract face from frame list
    parameter: frame_list[list]
    return: face_list[list]
    '''
    face_list = []
    for frame in frame_list:
        try:
            face = baseface(frame)
        except Exception:
            continue
        face_list.append(face)
    return face_list


def extract_face(frame):
    '''
    function: extract face from one frame
    parameter: frame
    return: face
    '''
    face = baseface(frame)
    return face


def extract_video(video_path):
    '''
    function: extract frames from one video
    parameter: video_path[str]
    return: frame_list[list], count[total frame number]
    '''
    videoCapture = cv2.VideoCapture(video_path)
    success = True
    count = 0
    frame_list = []
    while 1:
        success, frame = videoCapture.read()
        if success == False:
            break
        count = count+1
        frame_list.append(frame)
    return frame_list, count


def extract_mfcc(video_path, audio_path):
    '''
    function: extract audio mfcc
    parameter: video_path, audio_path
    return: mfcc[np, 13*121], audio[?]
    '''
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -loglevel quiet" %
               (video_path, audio_path))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(audio_path)
    mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc])
    return mfcc, audio


def save_npz(face_gt, mfcc_gt, identity1, identity5, video_path, success_num):
    save_path = video_path.replace(
        'lrw', 'lrw{}'.format(str(time.strftime("%m%d", time.localtime()))))
    save_path = '{}_{}.npz'.format(os.path.splitext(save_path)[0], success_num)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.savez(save_path, face_gt=face_gt, mfcc_gt=mfcc_gt,
             identity1=identity1, identity5=identity5)


def process_one_video(video_path):
    '''
    parameter: video_path
    return: face_gt, mfcc_gt, identity1, identity5[list]
    '''
    audio_path = '{}/{}.wav'.format(os.path.dirname(video_path)
                                    [0], os.path.splitext(video_path)[0])
    frame_list, frame_num = extract_video(video_path)
    mfcc, audio = extract_mfcc(video_path, audio_path)
    face_list = extract_face_list(frame_list)

    # print '1 frame - 0.04s    1 mfcc - 0.01s'
    # print 'total frame num: {}, total mfcc length: {}'.format(
    #     frame_num, mfcc.shape[1])
    if (float(len(audio))/16000) < (float(frame_num)/25):
        print(" *** WARNING: The audio (%.4fs) is shorter than the video (%.4fs). Type 'cont' to continue. *** " %
              (float(len(audio))/16000, float(frame_num)/25))

    mfcc_start = -20
    success_num = 0
    for i in range(4, frame_num, 5):
        mfcc_start += 20
        mfcc_end = mfcc_start+35
        if mfcc_end > mfcc.shape[1]:
            break
        try:
            '''face_gt'''
            face_gt = extract_face(frame_list[i])
            face_gt = cv2.resize(face_gt, (109, 109))
            '''mfcc_gt'''
            mfcc_gt = mfcc[1:, mfcc_start:mfcc_end]
            '''identity1 & identity5'''
            identity1 = random.sample(face_list, 1)
            identity1 = identity1[0]
            identity5 = random.sample(face_list, 5)
            identity5 = np.concatenate(identity5, axis=2)
            '''save to npz'''
            save_npz(face_gt, mfcc_gt, identity1,
                     identity5, video_path, success_num)
            success_num += 1
        except Exception, e:
            print traceback.print_exc()
            continue
    return success_num


def main_prepare_data(video_path_list):
    processed_num = 0
    total_num = len(video_path_list)
    for video_path in video_path_list:
        success_num = process_one_video(video_path)
        processed_num += 1
        print 'processed {}/{}, success {}'.format(
            processed_num, total_num, success_num)


def generate_path_txt(dataset_folder):
    print 'start generate txt'
    txt = open(LRW_TXT_PATH, 'w')
    video_path_list = []
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                video_path_list.append(video_path)
                txt.write(video_path+'\n')
    txt.close()
    print 'done generate txt, total number {}, saved in {}'.format(
        len(video_path_list), LRW_TXT_PATH)


if __name__ == "__main__":
    if args.device == 'mac':
        LRW_FOLDER = '../../../data/lrw/'
        LRW_TXT_PATH = './lrw_video_path.txt'
        DLIB_DAT_PATH = '../../../tools/shape_predictor_68_face_landmarks.dat'
        AVGLM_PATH = '../../../github/yousaidthat/data/avglm.mat'
        TFRECORDS_PATH = '../../../data/lrw1018/lipread_mp4/MIGHT/test/test.tfrecords'
        NPZ_PATH = '../../../data/lrw1018/lipread_mp4/MIGHT/test/'
    elif args.device == 'server009':
        LRW_FOLDER = '../../../data/lrw/'
        LRW_TXT_PATH = './lrw_video_path.txt'
        DLIB_DAT_PATH = '../../../tools/shape_predictor_68_face_landmarks.dat'
        AVGLM_PATH = '../../../github/yousaidthat/data/avglm.mat'
        TFRECORDS_PATH = '../../../data/lrw1016/lipread_mp4/MIGHT/test/test.tfrecords'
        NPZ_PATH = '../../../data/lrw1016/lipread_mp4/MIGHT/test/'
    elif args.device == 'ssh':
        LRW_FOLDER = '/workspace/liuhan/work/avasyn/data/lrw/'
        LRW_TXT_PATH = './lrw_video_path.txt'
        DLIB_DAT_PATH = '../../../../tools/shape_predictor_68_face_landmarks.dat'
        AVGLM_PATH = '../../../../github/yousaidthat/data/avglm.mat'
        TFRECORDS_PATH = '/workspace/liuhan/work/avasyn/src/face_lls/yousaidthat-tf/code02/lrw.tfrecords'
        NPZ_PATH = '/workspace/liuhan/work/avasyn/data/lrw1016/'
    if args.func == 'txt':
        generate_path_txt(LRW_FOLDER)
    elif args.func == 'data':
        dlibDetector = dlib.get_frontal_face_detector()
        dlibPredictor = dlib.shape_predictor(DLIB_DAT_PATH)
        U = scio.loadmat(AVGLM_PATH)
        U = U['avglm'][0][0][1][27:48]
        video_path_list = []
        txt = open(LRW_TXT_PATH, 'r')
        for line in txt.readlines():
            line = line.strip('\n')
            video_path_list.append(line)
        main_prepare_data(video_path_list)
    elif args.func == 'tfrecords':
        '''write tfrecords file'''
        print 'convert face data to tfrecords'
        data_writer(NPZ_PATH, TFRECORDS_PATH)
        print 'finish face data to tfrecords'

        '''read tfrecords file'''
        # read_and_decode(TFRECORDS_PATH, BATCH_SIZE, 5)

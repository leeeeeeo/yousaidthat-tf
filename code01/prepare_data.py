import cv2
import dlib
import os
import random
import glob
from scipy.io import wavfile
import subprocess
import numpy as np
import python_speech_features
from baseface import baseface
import traceback
import time


LRW_FOLDER = '../../../data/lrw/'


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
    print '\n'*1
    sample_rate, audio = wavfile.read(audio_path)
    mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = np.stack([np.array(i) for i in mfcc])
    return mfcc, audio


def save_npz(face_gt, mfcc_gt, identity1, identity5, video_path):
    save_path = video_path.replace(
        'lrw', 'lrw{}'.format(str(time.strftime("%m%d", time.localtime()))))
    save_path = '{}.npz'.format(os.path.splitext(save_path)[0])
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

    print '1 frame - 0.04s    1 mfcc - 0.01s'
    print 'total frame num: {}, total mfcc length: {}'.format(
        frame_num, mfcc.shape[1])
    if (float(len(audio))/16000) < (float(frame_num)/25):
        print(" *** WARNING: The audio (%.4fs) is shorter than the video (%.4fs). Type 'cont' to continue. *** " %
              (float(len(audio))/16000, float(frame_num)/25))

    mfcc_start = -20
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
            save_npz(face_gt, mfcc_gt, identity1, identity5, video_path)
            print '{}, frame: {}, mfcc_start: {}, mfcc_end: {}'.format(video_path,
                                                                       i, mfcc_start, mfcc_end)
        except Exception, e:
            print traceback.print_exc()
            continue


def main_prepare_data():
    video_path_list = []
    for root, dirs, files in os.walk(LRW_FOLDER):
        for file in files:
            if file.endswith('.mp4'):
                video_path_list.append(os.path.join(root, file))
    for video_path in video_path_list:
        process_one_video(video_path)


if __name__ == "__main__":
    main_prepare_data()

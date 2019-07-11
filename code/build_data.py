import cv2
import dlib
import os
import random
import scipy.io as scio
from scipy.io import wavfile
import subprocess
import numpy as np
import python_speech_features
import time
import argparse
from build_data_utils import data_writer, procrustes, dlibShape2Array
import soundfile
import traceback
import sys
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', help='which dataset to process?', type=str)
parser.add_argument('--device', help='which device?', type=str)
parser.add_argument(
    '--func', help='txt: generate dataset mp4 path txt, data: preprocess dataset, tfrecords: build tfrecords', default='data', type=str)
parser.add_argument(
    '--checkpoint', help='continue from checkpoint?',  type=int)
parser.add_argument('--npz', help='path to npz', type=str)
parser.add_argument('--tfrecords', help='path to tfrecords', type=str)
parser.add_argument('--matlab',  help='use matlab to generate mfcc?', type=int)
parser.add_argument(
    '--audionorm', help='mfcc normalize to [-1, 1] or [-25, 25]?', type=int)
parser.add_argument('--syncnet', help='use syncnet?', type=int)
parser.add_argument('--outputdir', help='output dir name', type=str)
parser.add_argument('--passnan', help='if nan, break', type=int)
args = parser.parse_args()
if args.matlab == 1:
    import matlab.engine
    eng = matlab.engine.start_matlab()

TFRECORDS_PATH = args.tfrecords
NPZ_PATH = args.npz
SYSTEM_ERROR_NUM = 0
SYSTEM_ERROR_FLAG = 0


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 2)


def baseface(dets, img):
    fleft = dets.left()
    ftop = dets.top()
    fwidth = dets.width()
    fheight = dets.height()
    d = dlib.rectangle(left=fleft, top=ftop, right=fwidth +
                       fleft, bottom=fheight+ftop)
    V = dlibPredictor(img, d)
    V = dlibShape2Array(V)[27:48]
    d, Z, transform = procrustes(U, V, scaling=True, reflection=False)
    T = np.array([[0, 0, 0], [0, 0, 0], [transform['translation']
                                         [0], transform['translation'][1], 1]])
    T[0:2, 0:2] = transform['rotation'] * transform['scale']
    T = T[:, 0:2].T
    dst = cv2.warpAffine(img, T, (240, 240))
    dst = dst[0:240, 0:240, :]
    dst = dst[22:22 + 197, 22:22 + 197, :]
    dst = cv2.resize(dst, (112, 112), interpolation=cv2.INTER_NEAREST)
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = dlibDetector(frame, 1)
            if len(dets) == 1:
                dets = dets[0]
                face = baseface(dets, frame)
            else:
                continue
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = dlibDetector(frame, 1)
    if len(dets) == 1:
        dets = dets[0]
        face = baseface(dets, frame)
        return 1, face
    else:
        return -1, None


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


def normalization_audio(x):
    return 1.0*(x-np.min(x))/(np.max(x)-np.min(x))


def extract_mfcc(video_path, audio_path):
    '''
    function: extract audio mfcc
    parameter: video_path, audio_path
    return: mfcc[np, 13*121], audio[?]
    '''
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -loglevel quiet" %
               (video_path, audio_path))
    # command = (
    #     "ffmpeg -threads 1 -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % ('../../syncnet_python/audio.wav', audio_path))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(audio_path)
    lowfreq = 300
    highfreq = 3700
    nfilt = 40
    mfcc = zip(*python_speech_features.mfcc(audio, sample_rate, nfilt=nfilt,
                                            lowfreq=lowfreq, highfreq=highfreq, winfunc=np.hamming))
    mfcc = np.stack([np.array(i) for i in mfcc])
    mfcc = mfcc[1:, :]
    if args.audionorm == 1:
        mfcc = normalization_audio(mfcc)
    elif args.audionorm == -1:
        mfcc = mfcc*1.0/abs(np.max(abs(mfcc)))
    elif args.audionorm == 25:
        mfcc = mfcc*25.0/abs(np.max(abs(mfcc)))
    else:
        mfcc = mfcc
    return mfcc, audio, 0


def extract_mfcc_matlab(video_path, audio_path):
    '''
    function: extract audio mfcc
    parameter: video_path, audio_path
    return: mfcc[np, 13*121], audio[?]
    '''
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -loglevel quiet" %
               (video_path, audio_path))
    # command = (
    #     "ffmpeg -threads 1 -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % ('../../syncnet_python/audio.wav', audio_path))
    output = subprocess.call(command, shell=True, stdout=None)
    audio, sample_rate = soundfile.read(audio_path)
    audio = np.reshape(audio, (audio.shape[0], 1))
    audio = matlab.double(audio.tolist())
    CC = eng.python_mfcc(audio)
    C = np.array(CC._data.tolist())
    shape = (CC.size[1], CC.size[0])
    C = C.reshape(shape).T
    C = C[1:, :]
    if args.audionorm == 1:
        C = np.nan_to_num(C)
        C = normalization_audio(C)
    elif args.audionorm == -1:
        C = np.nan_to_num(C)
        C = C*1.0/abs(np.max(abs(C)))
    elif args.audionorm == 25:
        C = np.nan_to_num(C)
        C = C * 25.0 / abs(np.max(abs(C)))
    else:
        C = C
    return C, audio, 1


def save_npz(face_blur, face_gt, mfcc_gt, identity1, identity5, video_path, success_num):
    save_path = video_path.replace(
        args.dataset, '{}_{}'.format(args.outputdir, str(time.strftime("%m%d", time.localtime()))))
    save_path = '{}_{}.npz'.format(os.path.splitext(save_path)[0], success_num)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.savez(save_path, face_blur=face_blur, face_gt=face_gt, mfcc_gt=mfcc_gt,
             identity1=identity1, identity5=identity5)


def process_one_video_syncnet(new_video_path, video_path):
    '''
    parameter: video_path
    return: face_gt, mfcc_gt, identity1, identity5[list]
    '''
    audio_path = '{}.wav'.format(os.path.splitext(new_video_path)[0])
    frame_list, frame_num = extract_video(new_video_path)
    if args.matlab == 1:
        mfcc, audio, isMATLAB = extract_mfcc_matlab(new_video_path, audio_path)
    elif args.matlab == 0:
        mfcc, audio, isMATLAB = extract_mfcc(new_video_path, audio_path)
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
            face_gt = cv2.resize(face_gt, (109, 109),
                                 interpolation=cv2.INTER_NEAREST)
            # face_gt = face_gt/(1.0*np.max(face_gt))
            face_blur = gaussian_blur(face_gt)
            # face_blur = face_blur/(1.0*np.max(face_blur))
            '''mfcc_gt'''
            mfcc_gt = mfcc[:, mfcc_start:mfcc_end]
            mfcc_gt = np.reshape(mfcc_gt, (12, 35, 1))
            '''identity1 & identity5'''
            identity1 = random.sample(face_list, 1)
            identity1 = identity1[0]
            # identity1 = identity1/(1.0*np.max(identity1))
            identity5 = random.sample(face_list, 5)
            identity5 = np.concatenate(identity5, axis=2)
            # identity5 = identity5/(1.0*np.max(identity5))
            '''save to npz'''
            save_npz(face_blur, face_gt, mfcc_gt, identity1,
                     identity5, video_path, success_num)
            success_num += 1
        except Exception, e:
            # print traceback.print_exc()
            continue
    return success_num, isMATLAB


def process_one_video_vox2(video_path):
    '''
    parameter: video_path
    return: face_gt, mfcc_gt, identity1, identity5[list]
    '''
    audio_path = '{}/{}.wav'.format(os.path.dirname(video_path)
                                    [0], os.path.splitext(video_path)[0])
    frame_list, frame_num = extract_video(video_path)
    if args.matlab == 1:
        mfcc, audio, isMATLAB = extract_mfcc_matlab(video_path, audio_path)
    elif args.matlab == 0:
        mfcc, audio, isMATLAB = extract_mfcc(video_path, audio_path)
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
            face_gt = cv2.resize(face_gt, (109, 109),
                                 interpolation=cv2.INTER_NEAREST)
            # face_gt = face_gt/(1.0*np.max(face_gt))
            face_blur = gaussian_blur(face_gt)
            # face_blur = face_blur/(1.0*np.max(face_blur))
            '''mfcc_gt'''
            mfcc_gt = mfcc[:, mfcc_start:mfcc_end]
            mfcc_gt = np.reshape(mfcc_gt, (12, 35, 1))
            '''identity1 & identity5'''
            identity1 = random.sample(face_list, 1)
            identity1 = identity1[0]
            # identity1 = identity1/(1.0*np.max(identity1))
            identity5 = random.sample(face_list, 5)
            identity5 = np.concatenate(identity5, axis=2)
            # identity5 = identity5/(1.0*np.max(identity5))
            '''save to npz'''
            save_npz(face_blur, face_gt, mfcc_gt, identity1,
                     identity5, video_path, success_num)
            success_num += 1
        except Exception, e:
            # print traceback.print_exc()
            continue
    return success_num, isMATLAB


def process_one_video(video_path):
    '''
    parameter: video_path
    return: face_gt, mfcc_gt, identity1, identity5[list]
    '''
    global SYSTEM_ERROR_FLAG
    global SYSTEM_ERROR_NUM
    audio_path = '{}/{}.wav'.format(os.path.dirname(video_path)
                                    [0], os.path.splitext(video_path)[0])
    frame_list, frame_num = extract_video(video_path)
    if args.matlab == 1:
        try:
            mfcc, audio, isMATLAB = extract_mfcc_matlab(video_path, audio_path)
            SYSTEM_ERROR_NUM = 0
            SYSTEM_ERROR_FLAG == 0
        except:
            if SYSTEM_ERROR_FLAG == 1:
                SYSTEM_ERROR_NUM += 1
                if SYSTEM_ERROR_NUM == 10:
                    print 'SYSTEM ERROR REACHED 10, NOW EXIT'
                    sys.exit()
            else:
                SYSTEM_ERROR_NUM = 1
            SYSTEM_ERROR_FLAG = 1
            print 'SYSTEM ERROR {}'.format(video_path)
            return 0, 1
    elif args.matlab == 0:
        mfcc, audio, isMATLAB = extract_mfcc(video_path, audio_path)
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
            success, face_gt = extract_face(frame_list[i])
            if success == -1:
                continue
            face_gt = cv2.resize(face_gt, (109, 109),
                                 interpolation=cv2.INTER_NEAREST)
            # face_gt = face_gt/(1.0*np.max(face_gt))
            face_blur = gaussian_blur(face_gt)
            # face_blur = face_blur/(1.0*np.max(face_blur))
            '''mfcc_gt'''
            mfcc_gt = mfcc[:, mfcc_start:mfcc_end]
            mfcc_gt = np.reshape(mfcc_gt, (12, 35, 1))
            '''identity1 & identity5'''

            if i - 25 < 0:
                lower_boundary = 0
            else:
                lower_boundary = i - 25
            if i + 25 < len(face_list):
                upper_boundary = i + 25
            else:
                upper_boundary = len(face_list)
                if upper_boundary - lower_boundary < 50:
                    if upper_boundary - 50 > 0:
                        lower_boundary = upper_boundary - 50
                    else:
                        lower_boundary = 0
            if len(face_list) < 5:
                continue

            identity1 = random.sample(
                face_list[lower_boundary:upper_boundary], 1)
            identity1 = identity1[0]
            # identity1 = identity1/(1.0*np.max(identity1))
            identity5 = random.sample(
                face_list[lower_boundary:upper_boundary], 5)
            identity5 = np.concatenate(identity5, axis=2)
            # identity5 = identity5/(1.0*np.max(identity5))
            '''save to npz'''
            save_npz(face_blur, face_gt, mfcc_gt, identity1,
                     identity5, video_path, success_num)
            success_num += 1
        except Exception, e:
            print traceback.print_exc()
            continue
    return success_num, isMATLAB


def syncnet(video_path):
    if not os.path.exists('syncnet_tmp/'):
        os.makedirs('syncnet_tmp/')
    command = ("rm -rf syncnet_tmp/*")
    output = subprocess.call(command, shell=True, stdout=None)
    command = (
        "python run_pipeline.py --videofile %s --reference name_of_video --data_dir syncnet_tmp/" % (video_path))
    output = subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'))
    command = (
        "python run_syncnet.py --videofile %s --reference name_of_video --data_dir syncnet_tmp/" % (video_path))
    output = subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'))
    command = (
        "python run_visualise.py --videofile %s --reference name_of_video --data_dir syncnet_tmp/" % (video_path))
    output = subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'))
    if os.path.exists('syncnet_tmp/pyavi/name_of_video/video_out.avi'):
        return 'syncnet_tmp/pyavi/name_of_video/video_out.avi'
    return None


def main_prepare_data_syncnet(processed_num, total_num, video_path_list, DATASET_CHECKPOINT_PATH):
    if processed_num == 0:
        total_num = len(video_path_list)
    for video_path in video_path_list:
        new_video_path = syncnet(video_path)
        if new_video_path != None:
            success_num, isMATLAB = process_one_video_syncnet(
                new_video_path, video_path)
            processed_num += 1
            checkpoint = '{},{},{}'.format(
                processed_num, total_num, video_path)
            txt = open(DATASET_CHECKPOINT_PATH, 'w')
            txt.write(checkpoint)
            txt.close()
            if isMATLAB == 1:
                print '{} {} MATLAB, processed {}/{}, success {}, {}'.format(datetime.now().strftime("%m/%d %H:%M:%S"), args.outputdir,
                                                                             processed_num, total_num, success_num, video_path)
            elif isMATLAB == 0:
                print '{} {} processed {}/{}, success {}, {}'.format(datetime.now().strftime("%m/%d %H:%M:%S"), args.outputdir,
                                                                     processed_num, total_num, success_num, video_path)
            command = ("rm -rf syncnet_tmp/*")
            output = subprocess.call(command, shell=True, stdout=None)
        else:
            continue


def main_prepare_data(processed_num, total_num, video_path_list, DATASET_CHECKPOINT_PATH):
    if processed_num == 0:
        total_num = len(video_path_list)
    for video_path in video_path_list:
        success_num, isMATLAB = process_one_video(video_path)
        processed_num += 1
        checkpoint = '{},{},{}'.format(processed_num, total_num, video_path)
        txt = open(DATASET_CHECKPOINT_PATH, 'w')
        txt.write(checkpoint)
        txt.close()
        if isMATLAB == 1:
            print '{} {} MATLAB, processed {}/{}, success {}, {}'.format(datetime.now().strftime("%m/%d %H:%M:%S"), args.outputdir,
                                                                         processed_num, total_num, success_num, video_path)
        elif isMATLAB == 0:
            print '{} {} processed {}/{}, success {}, {}'.format(datetime.now().strftime("%m/%d %H:%M:%S"), args.outputdir,
                                                                 processed_num, total_num, success_num, video_path)


def generate_path_txt(dataset_folder):
    print 'start generate txt'
    if not os.path.exists(os.path.dirname(DATASET_TXT_PATH)):
        os.makedirs(os.path.dirname(DATASET_TXT_PATH))
    txt = open(DATASET_TXT_PATH, 'w')
    video_path_list = []
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                video_path_list.append(video_path)
    random.shuffle(video_path_list)
    for video_path in video_path_list:
        txt.write(video_path+'\n')
    txt.close()
    print 'done generate txt, total number {}, saved in {}'.format(
        len(video_path_list), DATASET_TXT_PATH)



# def test_base_face():
#     INPUT_IMAGE_PATH = '/Users/lls/Documents/face/data/trump_12.png'
#     img = cv2.imread(INPUT_IMAGE_PATH)
#     success, face = extract_face(img)
#     face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
#     cv2.imshow('a',face)
#     cv2.waitKey(0)
    
    

if __name__ == "__main__":
    if args.device == 'mac':
        if args.dataset == 'lrw':
            DATASET_FOLDER = '../../../data/lrw/'
            DATASET_TXT_PATH = '../txt/lrw_video_path_mac.txt'
            DATASET_CHECKPOINT_PATH = '../txt/lrw_checkpoint_mac.txt'
        elif args.dataset == 'voxceleb2':
            DATASET_FOLDER = '../../../data/voxceleb2/'
            DATASET_TXT_PATH = '../txt/voxceleb2_video_path_mac.txt'
            DATASET_CHECKPOINT_PATH = '../txt/voxceleb2_checkpoint_mac.txt'
        DLIB_DAT_PATH = '../../../tools/shape_predictor_68_face_landmarks.dat'
        AVGLM_PATH = '../../../github/yousaidthat/data/avglm.mat'
    elif args.device == 'server009':
        if args.dataset == 'lrw-full':
            DATASET_FOLDER = '/media/server009/data/dataset/lrw-full/'
            DATASET_TXT_PATH = '/media/server009/data/dataset/lrw-full/lrw-full_video_path_server009.txt'
            DATASET_CHECKPOINT_PATH = '/media/server009/data/dataset/lrw-full/{}_checkpoint_server009.txt'.format(
                args.outputdir)
        elif args.dataset == 'voxceleb2':
            DATASET_FOLDER = '/media/server009/data/dataset/voxceleb2/'
            DATASET_TXT_PATH = '/media/server009/data/dataset/voxceleb2/voxceleb2_video_path_server009.txt'
            DATASET_CHECKPOINT_PATH = '/media/server009/data/dataset/voxceleb2/{}_checkpoint_server009.txt'.format(
                args.outputdir)
        elif args.dataset == 'lrs2':
            DATASET_FOLDER = '/media/server009/data/dataset/lrs2/'
            DATASET_TXT_PATH = '/media/server009/data/dataset/lrs2/lrs2_video_path_server009.txt'
            DATASET_CHECKPOINT_PATH = '/media/server009/data/dataset/lrs2/{}_checkpoint_server009.txt'.format(
                args.outputdir)
        elif args.dataset == 'lrs3':
            DATASET_FOLDER = '/media/server009/data/dataset/lrs3/'
            DATASET_TXT_PATH = '/media/server009/data/dataset/lrs3/lrs3_video_path_server009.txt'
            DATASET_CHECKPOINT_PATH = '/media/server009/data/dataset/lrs3/{}_checkpoint_server009.txt'.format(
                args.outputdir)
        DLIB_DAT_PATH = '/media/server009/seagate/liuhan/lls/github/yousaidthat/model/landmark_68.dat'
        AVGLM_PATH = '../../../github/yousaidthat/data/avglm.mat'
    elif args.device == 'ssh':
        if args.dataset == 'lrw':
            DATASET_FOLDER = '/workspace/liuhan/work/avasyn/data/lrw/'
            DATASET_TXT_PATH = '/workspace/liuhan/work/avasyn/data/lrw/lrw_video_path_ssh.txt'
            DATASET_CHECKPOINT_PATH = '/workspace/liuhan/work/avasyn/data/lrw/{}_checkpoint_ssh.txt'.format(
                args.outputdir)
        elif args.dataset == 'voxceleb2':
            DATASET_FOLDER = '/workspace/liuhan/work/avasyn/data/voxceleb2/'
            DATASET_TXT_PATH = '/workspace/liuhan/work/avasyn/data/voxceleb2/voxceleb2_video_path_ssh.txt'
            DATASET_CHECKPOINT_PATH = '/workspace/liuhan/work/avasyn/data/voxceleb2/{}_checkpoint_ssh.txt'.format(
                args.outputdir)
        elif args.dataset == 'lrs2':
            DATASET_FOLDER = '/workspace/liuhan/work/avasyn/data/lrs2/'
            DATASET_TXT_PATH = '/workspace/liuhan/work/avasyn/data/lrs2/lrs2_video_path_ssh.txt'
            DATASET_CHECKPOINT_PATH = '/workspace/liuhan/work/avasyn/data/lrs2/{}_checkpoint_ssh.txt'.format(
                args.outputdir)
        elif args.dataset == 'lrs3':
            DATASET_FOLDER = '/workspace/liuhan/work/avasyn/data/lrs3/'
            DATASET_TXT_PATH = '/workspace/liuhan/work/avasyn/data/lrs3/lrs3_video_path_ssh.txt'
            DATASET_CHECKPOINT_PATH = '/workspace/liuhan/work/avasyn/data/lrs3/{}_checkpoint_ssh.txt'.format(
                args.outputdir)
        DLIB_DAT_PATH = '/workspace/liuhan/work/avasyn/tools/shape_predictor_68_face_landmarks.dat'
        AVGLM_PATH = '/workspace/liuhan/work/avasyn/github/yousaidthat/data/avglm.mat'
    elif args.device == 'ssh7':
        if args.dataset == 'lrw-full':
            DATASET_FOLDER = '/media/liuhan/liuhan_4T/dataset/lrw-full/'
            DATASET_TXT_PATH = '/media/liuhan/liuhan_4T/dataset/lrw-full/lrw-full_video_path_ssh7.txt'
            DATASET_CHECKPOINT_PATH = '/media/liuhan/liuhan_4T/dataset/lrw-full/{}_checkpoint_ssh7.txt'.format(
                args.outputdir)
        elif args.dataset == 'voxceleb2':
            DATASET_FOLDER = '/media/liuhan/liuhan_4T/dataset/voxceleb2/'
            DATASET_TXT_PATH = '/media/liuhan/liuhan_4T/dataset/voxceleb2/voxceleb2_video_path_ssh.txt'
            DATASET_CHECKPOINT_PATH = '/media/liuhan/liuhan_4T/dataset/voxceleb2/{}_checkpoint_ssh.txt'.format(
                args.outputdir)
        elif args.dataset == 'lrs2':
            DATASET_FOLDER = '/media/liuhan/liuhan_4T/dataset/lrs2/'
            DATASET_TXT_PATH = '/media/liuhan/liuhan_4T/dataset/lrs2/lrs2_video_path_ssh7.txt'
            DATASET_CHECKPOINT_PATH = '/media/liuhan/liuhan_4T/dataset/lrs2/{}_checkpoint_ssh7.txt'.format(
                args.outputdir)
        elif args.dataset == 'lrs3':
            DATASET_FOLDER = '/media/liuhan/liuhan_4T/dataset/lrs3/'
            DATASET_TXT_PATH = '/media/liuhan/liuhan_4T/dataset/lrs3/lrs3_video_path_ssh7.txt'
            DATASET_CHECKPOINT_PATH = '/media/liuhan/liuhan_4T/dataset/lrs3/{}_checkpoint_ssh7.txt'.format(
                args.outputdir)
        DLIB_DAT_PATH = '../../yousaidthat/model/landmark_68.dat'
        AVGLM_PATH = '../../yousaidthat/data/avglm.mat'
    if args.func == 'txt':
        generate_path_txt(DATASET_FOLDER)
    elif args.func == 'data':
        dlibDetector = dlib.get_frontal_face_detector()
        dlibPredictor = dlib.shape_predictor(DLIB_DAT_PATH)
        U = scio.loadmat(AVGLM_PATH)
        U = U['avglm'][0][0][1][27:48]
        video_path_list = []
        if args.checkpoint == 1:
            checkpoint_txt = open(DATASET_CHECKPOINT_PATH, 'r')
            checkpoint = checkpoint_txt.readline().split(',')
            checkpoint = (checkpoint[0], checkpoint[1], checkpoint[2])
            processed_num = int(checkpoint[0])
            total_num = int(checkpoint[1])
            checkpoint = checkpoint[2]
            txt = open(DATASET_TXT_PATH, 'r')
            txt = [line.strip('\n') for line in txt.readlines()]
            video_path_list = txt[txt.index(checkpoint):]
        elif args.checkpoint == 0:
            processed_num = 0
            total_num = 0
            txt = open(DATASET_TXT_PATH, 'r')
            for line in txt.readlines():
                line = line.strip('\n')
                video_path_list.append(line)
        if args.syncnet == 0:
            main_prepare_data(processed_num, total_num,
                              video_path_list, DATASET_CHECKPOINT_PATH)
        elif args.syncnet == 1:
            main_prepare_data_syncnet(processed_num, total_num,
                                      video_path_list, DATASET_CHECKPOINT_PATH)
        # test_base_face()
    elif args.func == 'tfrecords':
        '''write tfrecords file'''
        print 'START {}'.format(TFRECORDS_PATH)
        data_writer(NPZ_PATH, TFRECORDS_PATH, args.passnan)
        print 'END {}'.format(TFRECORDS_PATH)

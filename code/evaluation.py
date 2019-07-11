import tensorflow as tf
import argparse
import cv2
import os
import numpy as np
np.set_printoptions(threshold='nan')
import python_speech_features
from scipy.io import wavfile
import subprocess

import dlib
import scipy.io as scio
from build_data_utils import dlibShape2Array, procrustes
from PIL import Image, ImageDraw, ImageFont
import soundfile
import time
from time import sleep
from skimage import io


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str)
parser.add_argument('--face_detection', type=int)
parser.add_argument('--gpu', type=str)
parser.add_argument('--bn', help='use bn?', type=int)
parser.add_argument('--audio', help='use audio?', type=int)
parser.add_argument('--audiofc', help='use audio fc?', type=int)
parser.add_argument('--face', help='use identity?', type=int)
parser.add_argument('--facefc', help='use identity fc?', type=int)
parser.add_argument('--decoder', help='use decoder?', type=int)
parser.add_argument('--lip', help='use lip?', type=int)
parser.add_argument(
    '--avoidlayers', help='layers do not need initialization', nargs='+')
parser.add_argument('--momentum', help='momentum', type=float)
parser.add_argument('--ckpt', type=str)
parser.add_argument('--matlab', type=int)
parser.add_argument('--idnum', type=int)
parser.add_argument('--audionorm', type=int)
parser.add_argument('--output', type=str)
parser.add_argument('--mode', type=str)
parser.add_argument('--mp4', help='generate mp4?', type=int)
parser.add_argument(
    '--xavier', help='use xavier_initializer or truncated_normal_initializer', type=int)
parser.add_argument('--audiopath', help='path to audio',
                    default='../../syncnet_python/audio.wav', type=str)

# parser.add_argument('--meta', help='path to meta file', type=str)
# parser.add_argument('--data', help='path to data file', type=str)
# parser.add_argument(
#     '--checkpointdir', help='path to checkpoint dir', type=str)
parser.add_argument(
    '--images', help='path to images path', type=str)
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.matlab == 1:
    import matlab.engine
    eng = matlab.engine.start_matlab()


CHECKPOINT_DIR = args.ckpt
FIVE_IMAGES_PATH = args.images

TMP_WAV_PATH = 'tmp.wav'
WAV_TO_GENERATE_MP4_PATH = 'mp4_audio.wav'
OUTPUT_DIR = 'output/{}'.format(args.output)
BATCH_SIZE = 1
USE_BN = args.bn
USE_AUDIO = args.audio
USE_LIP = args.lip
USE_DECODER = args.decoder
USE_FACE = args.face
USE_FACEFC = args.facefc
USE_AUDIOFC = args.audiofc
USE_XAVIER = args.xavier
AUDIO_PATH = args.audiopath
AVOID_LAYERS_LIST = args.avoidlayers
# CHECKPOINT_DIR = arg.checkpointdir
# META_PATH = arg.meta
# DATA_PATH = arg.data
# FIVE_IMAGES_PATH = arg.images


def baseface(img):
    dets = dlibDetector(img, 1)[0]
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


def read_five_images(five_images_path, idnum, face_detection):
    identity_list = []
    for root, dirs, files in os.walk(five_images_path):
        for file in files:
            img_path = os.path.join(root, file)
            face = io.imread(img_path)
            if face_detection == 1:
                face = baseface(face)
            # face = face/(1.0*np.max(face))
            identity_list.append(face)
    if idnum == 1:
        identity1_0 = np.reshape(identity_list[0], (1, 112, 112, 3))
        return identity1_0
    elif idnum == 5:
        assert len(identity_list) == 5
        identity5_0 = np.reshape(np.concatenate(
            identity_list, axis=2), (1, 112, 112, 15))
        return identity5_0


def normalization_audio(x):
    return 1.0*(x-np.min(x))/(np.max(x)-np.min(x))


def extract_mfcc(audio_path):
    command = ("ffmpeg -threads 1 -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" %
               (audio_path, TMP_WAV_PATH))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(TMP_WAV_PATH)
    audio_pad = np.zeros((16000,), dtype=int)
    audio_list = [audio_pad, audio, audio_pad]
    audio = np.concatenate(audio_list)
    lowfreq = 300
    highfreq = 3700
    nfilt = 40
    mfcc = zip(*python_speech_features.mfcc(audio, samplerate=sample_rate,  nfilt=nfilt,
                                            lowfreq=lowfreq, highfreq=highfreq, winfunc=np.hamming))
    C = np.stack([np.array(i) for i in mfcc])
    if args.audionorm == -1:
        C = np.nan_to_num(C)
        C = C*1.0/abs(np.max(abs(C)))
    elif args.audionorm == 1:
        C = np.nan_to_num(C)
        C = normalization_audio(C)
    elif args.audionorm == 25:
        C = np.nan_to_num(C)
        C = C*25.0/abs(np.max(abs(C)))
    return C


def extract_mfcc_without_hamming(audio_path):
    command = ("ffmpeg -threads 1 -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" %
               (audio_path, TMP_WAV_PATH))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(TMP_WAV_PATH)
    audio_pad = np.zeros((16000,), dtype=int)
    audio_list = [audio_pad, audio, audio_pad]
    audio = np.concatenate(audio_list)
    lowfreq = 300
    highfreq = 3700
    nfilt = 40
    mfcc = zip(*python_speech_features.mfcc(audio, samplerate=sample_rate))
    C = np.stack([np.array(i) for i in mfcc])
    if args.audionorm == 1:
        C = np.nan_to_num(C)
        C = C*1.0/abs(np.max(abs(C)))
    elif args.audionorm == -1:
        C = np.nan_to_num(C)
        C = normalization_audio(C)
    elif args.audionorm == 25:
        C = np.nan_to_num(C)
        C = C*25.0/abs(np.max(abs(C)))
    return C


def extract_mfcc_matlab(audio_path):
    '''
    function: extract audio mfcc
    parameter: video_path, audio_path
    return: mfcc[np, 13*121], audio[?]
    '''
    # command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s -loglevel quiet" %
    #            (video_path, audio_path))
    command = (
        "ffmpeg -threads 1 -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, TMP_WAV_PATH))
    output = subprocess.call(command, shell=True, stdout=None)
    audio, sample_rate = soundfile.read(TMP_WAV_PATH)
    audio_pad = np.zeros((16000,), dtype=np.float64)
    audio_list = [audio_pad, audio, audio_pad]
    audio = np.concatenate(audio_list)
    audio = np.reshape(audio, (audio.shape[0], 1))
    soundfile.write(WAV_TO_GENERATE_MP4_PATH, audio, sample_rate)
    audio = matlab.double(audio.tolist())
    CC = eng.python_mfcc(audio)
    C = np.array(CC._data.tolist())
    shape = (CC.size[1], CC.size[0])
    C = C.reshape(shape).T
    if args.audionorm == 1:
        C = np.nan_to_num(C)
        C = C*1.0/abs(np.max(abs(C)))
    elif args.audionorm == -1:
        C = np.nan_to_num(C)
        C = normalization_audio(C)
    elif args.audionorm == 25:
        # C = np.nan_to_num(C)
        # C = C*25.0/abs(np.max(abs(C)))
        C = C
    return C


def save_video(output_dir, wav_path, save_path):
    # pic + audio 1
    if args.device == 'server009':
        command = ("ffmpeg -y -r 25 -i {}/%03d.jpg -i {} -vf \" pad=ceil(iw/2)*2:ceil(ih/2)*2\" -strict -2 {}".format(
            output_dir, wav_path, save_path))
    elif args.device == 'ssh7':
        command = ("ffmpeg -y -r 25 -i {}/%03d.jpg -i {} -vf \" pad=ceil(iw/2)*2:ceil(ih/2)*2\" -strict -2 -ar 44100 {}".format(
            output_dir, wav_path, save_path))
    # pic + audio 2
    # command = ("ffmpeg -threads 2 -y -r 25 -i {}/%03d.jpg -i {} -vf \" pad=ceil(iw/2)*2:ceil(ih/2)*2\" -absf aac_adtstoasc -strict -2  {}".format(
    #     output_dir, wav_path, save_path))
    # only pic
    # command = (
    #     "ffmpeg -y -loop 0 -f image2 -i {}/%03d.jpg -vcodec libx264 -vf \" pad=ceil(iw/2)*2:ceil(ih/2)*2\" -t 10 -r 25  {}".format(output_dir, save_path))
    output = subprocess.call(command, shell=True, stdout=None)


def evaluation():
    identity5_0 = read_five_images(
        FIVE_IMAGES_PATH, args.idnum, face_detection=args.face_detection)
    if args.idnum == 5:
        identity = tf.placeholder(tf.float32, shape=[1, 112, 112, 15])
    elif args.idnum == 1:
        identity = tf.placeholder(tf.float32, shape=[1, 112, 112, 3])
    mfcc = tf.placeholder(tf.float32, shape=[1, 12, 35, 1])
    if args.matlab == 1:
        C = extract_mfcc_matlab(AUDIO_PATH)
    else:
        C = extract_mfcc(AUDIO_PATH)
    C_length = 0
    save_path_list = []
    for j in range(0, C.shape[1]-34, 4):
        C_length = C_length + 1
    count = 0
    if args.mode == 'upsample':
        import speech2vid_inference_finetune
        speech2vid = speech2vid_inference_finetune.Speech2Vid(
            USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, args.idnum)
    elif args.mode == 'deconv':
        import speech2vid_inference_deconv
        speech2vid = speech2vid_inference_deconv.Speech2Vid(
            USE_AUDIO, USE_BN, USE_LIP, USE_DECODER, USE_FACEFC, USE_AUDIOFC, USE_FACE, USE_XAVIER, AVOID_LAYERS_LIST, args.idnum)

    audio_encoder_output = speech2vid.audio_encoder(
        mfcc)
    identity_encoder_output, x7_face, x4_face, x3_face = speech2vid.identity_encoder(
        identity, args.idnum)
    prediction = speech2vid.image_decoder(
        audio_encoder_output, identity_encoder_output, x7_face, x4_face, x3_face)

    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load checkpoint success, global_step is %s' % global_step)
        else:
            print 'load checkpoint fail'

        for j in range(0, C.shape[1] - 34, 4):
            start_time = time.time()
            mfcc_0 = C[1:, j:j+35]
            mfcc_0 = np.reshape(mfcc_0, (1, 12, 35, 1))
            prediction_, identity_encoder_output_, audio_encoder_output_ = sess.run([prediction, identity_encoder_output, audio_encoder_output], feed_dict={
                identity: identity5_0, mfcc: mfcc_0})
            '''display tensor value'''
            # # for t in ['relu7_audio/Relu']:
            # # for t in ['concat8/concat8', 'upsamp1_1/resampler/Resampler', 'upsamp1_2/resampler/Resampler', 'upsamp1_3/resampler/Resampler', 'upsamp2/resampler/Resampler']:
            # for t in ['upsamp1_2/ResizeNearestNeighbor']:
            #     # for t in ['concat8/concat8', 'conv8/BiasAdd', 'relu8/Relu', 'upsamp1_1/ResizeNearestNeighbor', 'conv1_1/BiasAdd','relu1_1/Relu','upsamp1_2/ResizeNearestNeighbor','conv1_2/BiasAdd','relu1_2/Relu','upsamp1_3/ResizeNearestNeighbor','conv1_3/BiasAdd','relu1_3/Relu','upsamp2/ResizeNearestNeighbor','conv2/BiasAdd','relu2/Relu','upsamp3_1/ResizeNearestNeighbor','conv3_1/BiasAdd','relu3_1/Relu','conv3_2/BiasAdd','relu3_2/Relu','upsamp4/ResizeNearestNeighbor','conv4/BiasAdd','relu4/Relu','upsamp5_1/ResizeNearestNeighbor','conv5_1/BiasAdd','relu5_1/Relu','conv5_2/BiasAdd','relu5_2/Relu','conv5_3/BiasAdd']:
            #     tensor = sess.graph.get_tensor_by_name(
            #         "{}:0".format(t))
            #     tensor_ = sess.run(tensor, feed_dict={
            #         identity: identity5_0, mfcc: mfcc_0})
            #     # print t, np.mean(tensor_)
            #     # print tensor_
            #     for i in range(tensor_.shape[3]):
            #         print t,i+1,np.mean(tensor_[:,:,:,i]),np.var(tensor_[:,:,:,i])
            '''save tensor value to txt'''
            # tensor_name_list = [
            #     tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            # for tensor_name in tensor_name_list:
            #     print tensor_name
            #     if 'Place' not in tensor_name and 'save' not in tensor_name and 'Pad' not in tensor_name and 'Conv2D' not in tensor_name:
            #         tensor = sess.graph.get_tensor_by_name(
            #             "{}:0".format(tensor_name))
            #         tensor_ = sess.run(tensor)
            #         print type(tensor_)
            #         txt_path = 'tmp/{}.npy'.format(
            #             tensor_name.replace('/', '-'))
            #         np.save(txt_path, tensor_)
            # break
            prediction_Image = np.squeeze(prediction_)
            prediction_Image = 255*prediction_Image/np.max(prediction_Image)
            prediction_Image = Image.fromarray(
                prediction_Image.astype(np.uint8))
            '''put text'''
            text = str(count)
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
            draw = ImageDraw.Draw(prediction_Image)
            draw.text((10, 10), text=text, font=font)
            # prediction.show()
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            count = count+1
            save_path = ('%s/%03d.jpg' % (OUTPUT_DIR, count))
            save_path_list.append(save_path)
            prediction_Image.save(save_path)
            print 'Finish {}/{}, Audio output mean {}, Identity output mean {}, Prediction mean {}, Elapsed time {}'.format(
                count, C_length, np.mean(audio_encoder_output_), np.mean(identity_encoder_output_), np.mean(prediction_),   time.time()-start_time)

    if args.mp4 == 1:
        save_video(OUTPUT_DIR, WAV_TO_GENERATE_MP4_PATH,
                   '{}-{}.mp4'.format(OUTPUT_DIR, global_step))


'''extract weights from ckpt'''
# txt = open('../variable_name.txt', 'w')
# reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
# all_variables = reader.get_variable_to_shape_map()
# for key in all_variables:
#     print key
#     txt.write(key+'\n')
#     # if np.isnan(np.mean(reader.get_tensor(key))):
#     #     print '{}, {}'.format(key, np.mean(reader.get_tensor(key)))
# txt.close()
'''extract tensor names from ckpt'''
# txt = open('../tensor_name.txt', 'w')
# tensor_name_list = [
#     tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
#     print tensor_name
#     txt.write(tensor_name+'\n')
#     # if np.isnan(np.mean(reader.get_tensor(key))):
#     #     print '{}, {}'.format(key, np.mean(reader.get_tensor(key)))
# txt.close()
'''save tensor value to mat'''
# tensor_value_dict = {}
# tensor_name_list = [
#     tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
#     if 'Place' not in tensor_name and 'save' not in tensor_name:
#         tensor = sess.graph.get_tensor_by_name(
#             "{}:0".format(tensor_name))
#         tensor_ = sess.run(tensor)

#         tensor_value_dict['{}'.format(
#             tensor_name)] = tensor_
# scio.savemat('../tensor_value.mat', tensor_value_dict)
'''save tensor value to npz'''
# tensor_value_dict = {}
# tensor_name_list = [
#     tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
#     if 'Place' not in tensor_name and 'save' not in tensor_name:
#         tensor = sess.graph.get_tensor_by_name(
#             "{}:0".format(tensor_name))
#         tensor_ = sess.run(tensor)

#         tensor_value_dict['{}'.format(
#             tensor_name)] = tensor_
# np.savez('../tensor_value.npz', tensor_value_dict)


if __name__ == '__main__':
    if args.device == 'mac':
        DLIB_DAT_PATH = '../../../tools/shape_predictor_68_face_landmarks.dat'
        AVGLM_PATH = '../../../github/yousaidthat/data/avglm.mat'
    elif args.device == 'server009':
        DLIB_DAT_PATH = '../../../tools/shape_predictor_68_face_landmarks.dat'
        AVGLM_PATH = '../../../github/yousaidthat/data/avglm.mat'
    elif args.device == 'ssh':
        DLIB_DAT_PATH = '../../../../tools/shape_predictor_68_face_landmarks.dat'
        AVGLM_PATH = '../../../../github/yousaidthat/data/avglm.mat'
    elif args.device == 'ssh7':
        DLIB_DAT_PATH = '../../yousaidthat/model/landmark_68.dat'
        AVGLM_PATH = '../../yousaidthat/data/avglm.mat'
    dlibDetector = dlib.get_frontal_face_detector()
    dlibPredictor = dlib.shape_predictor(DLIB_DAT_PATH)
    U = scio.loadmat(AVGLM_PATH)
    U = U['avglm'][0][0][1][27:48]
    evaluation()

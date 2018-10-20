import dlib
import cv2
import scipy.io as scio
import numpy as np
import os


DLIB_DAT_PATH = '../../../tools/shape_predictor_68_face_landmarks.dat'
AVGLM_PATH = '../../../github/yousaidthat/data/avglm.mat'


dlibDetector = dlib.get_frontal_face_detector()
dlibPredictor = dlib.shape_predictor(DLIB_DAT_PATH)
U = scio.loadmat(AVGLM_PATH)
U = U['avglm'][0][0][1][27:48]


def procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)), 0)
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection is not 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    if my < m:
        T = T[:my, :]
    c = muX - b*np.dot(muY, T)
    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform


def dlibShape2Array(landmark2D, dtype="int"):
    coords = []
    for i in range(0, 68):
        coords.append([int(landmark2D.part(i).x), int(landmark2D.part(i).y)])
    return np.asarray(coords)


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


def process_casia():
    for root, dirs, files in os.walk('/media/server009/seagate/dataset/CASIA/CASIA-WebFace/'):
        for file in files:
            imgPath = os.path.join(root, file)
            print imgPath
            savePath = '/media/server009/seagate/dataset/CASIA/CASIA-WebFace-align/{}-{}'.format(
                imgPath.split('/')[-2], imgPath.split('/')[-1])
            img = cv2.imread(imgPath)
            try:
                aligned_face = baseface(img)
                cv2.imwrite(savePath, aligned_face)
            except:
                continue


def mainBaseface():
    img = cv2.imread('../../../github/yousaidthat/data/obama.jpg')
    dst = baseface(img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)


if __name__ == "__main__":
    mainBaseface()

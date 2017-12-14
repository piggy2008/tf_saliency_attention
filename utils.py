import numpy as np
import os
import cv2
import scipy.io as sio
import struct
from matplotlib import pyplot as plt

config_CRF_CRF_DIR = 'densecrf/prog_refine_pascal_v4'
config_CRF_im_sz = 514
config_CRF_CRF_ITER = 3
config_CRF_px = 3
config_CRF_py = 3
config_CRF_pw = 3
config_CRF_bx = 50
config_CRF_by = 50
config_CRF_br = 3
config_CRF_bg = 3
config_CRF_bb = 3
config_CRF_bw = 5


def preprocess(image, prior, input_shape=512):
    x = np.array(image, dtype=np.float32)
    x = x[:, :, ::-1]
    mean = (104.00699, 116.66877, 122.67892)
    mean = np.array(mean, dtype=np.float32)
    x = (x - mean)
    w, h, _ = x.shape
    prior_arr = np.array(prior, dtype=np.float32)
    input = np.zeros([input_shape, input_shape, 4], dtype=np.float32)
    input[:w, :h, :3] = x
    input[:w, :h, 3] = prior_arr


    return input[np.newaxis, ...], input[np.newaxis, :, :, :3]

def crf_refine(imgfilename, img, inference_map):

    if not os.path.exists('.ppmimg'):
        os.mkdir('.ppmimg')

    if not os.path.exists('.mat_dcl'):
        os.mkdir('.mat_dcl')

    if not os.path.exists('.dcl_crf'):
        os.mkdir('.dcl_crf')

    # np.transpose(inference_map)
    cv2.imwrite(os.path.join('.ppmimg', imgfilename + '.ppm'), img)
    data = np.zeros([config_CRF_im_sz, config_CRF_im_sz, 2], dtype=np.float16)
    data[:inference_map.shape[0], :inference_map.shape[1], 0] = inference_map
    data[:inference_map.shape[0], :inference_map.shape[1], 1] = inference_map


    sio.savemat(os.path.join('.mat_dcl', imgfilename + '_blob_0.mat'), {'data': data})

    CRF_CMD = '%s -id .ppmimg -fd .mat_dcl -sd .dcl_crf -i %d -px %d ' \
              '-py %d -pw %d -bx %d -by %d -br %d -bg %d -bb %d -bw %d' %(config_CRF_CRF_DIR,
    config_CRF_CRF_ITER, config_CRF_px, config_CRF_py, config_CRF_pw, config_CRF_bx,
    config_CRF_by, config_CRF_br, config_CRF_bg, config_CRF_bb, config_CRF_bw)

    os.system(CRF_CMD)

def loadBinFile(file_path):
    file = open(file_path, 'rb')

    row, = struct.unpack('i', file.read(4))
    col, = struct.unpack('i', file.read(4))
    channel, = struct.unpack('i', file.read(4))

    total = row * col * channel
    data = np.zeros([1, total])
    for i in range(0, total):
        data[0, i],  = struct.unpack('f', file.read(4))

    out = data.reshape([row, col, channel])
    out = np.squeeze(out)
    out = (out - np.amin(out)) / (np.amax(out) - np.amin(out) + 0.00001)
    print out


if __name__ == '__main__':
    img = cv2.imread('Comp_195.bmp')
    anno = cv2.imread('Comp_195.png', 0)
    crf_refine('test', img, anno)
    # loadBinFile('.dcl_crf/test.bin')
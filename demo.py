from model import VideoSailency
import tensorflow as tf
from moviepy.editor import VideoClip
import os
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

test_dir = '/home/ty/data/davis/davis_test'
test_prior_dir = '/home/ty/data/davis/davis_flow_prior'
list_file_path = '/home/ty/data/davis/davis_test_seq2.txt'

result_dir = 'total_result/result_rnn_crf_threshold_2018-07-16 10:29:07'

list_file = open(list_file_path)
test_names = [line.strip() for line in list_file]

color = [0, 0, 0]
i = 0

fps = 24
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
vw = cv2.VideoWriter('show.avi', fourcc, fps, (1020, 287))

while(True):

    images_path = test_names[i].split(',')
    image_path = os.path.join(test_dir, images_path[3] + '.jpg')
    image = cv2.imread(image_path)

    sal_result = os.path.join(result_dir, images_path[3] + '.png')
    result = cv2.imread(sal_result)


    image = cv2.resize(image, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_LINEAR)

    constant = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)
    constant2 = cv2.copyMakeBorder(result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)

    shows = cv2.hconcat((image, result))
    vw.write(shows)

    # cv2.imshow('video', shows)
    # cv2.waitKey(1)
    i += 1
    if i == len(test_names) - 1:
        i = 0
        vw.release()
        break

# import tensorflow as tf
# import numpy as np
# from matplotlib import pyplot as plt
# # Function in python
# def dummy(input_center):
#     sigma = 0.25
#     x = np.arange(0, 1, 0.001953125)
#     y = np.arange(0, 1, 0.001953125)
#     x, y = np.meshgrid(x, y)
#     z_batch = np.zeros([4, 512, 512, 1], dtype=np.float32)
#     for i in range(input_center.shape[0]):
#         if (input_center[i] > [0, 0]).all() and (input_center[i] < [1, 1]).all():
#             z = np.exp(-((x - input_center[i, 0]) ** 2 + (y - input_center[i, 1]) ** 2) / (sigma ** 2))
#         else:
#             z = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (sigma ** 2))
#
#         z = z[..., np.newaxis]
#         z_batch[i] = z
#     print('---------------')
#     print(input_center.shape)
#     # a = input_center[0] + input_center[1]
#     # plt.figure()
#     #
#     # plt.imshow(z, plt.cm.gray)
#     # plt.show()
#     return z
#
# input = [[0.1, 0.2], [0.4, 0.5], [-0.4, 0.5], [0.4, 0.5]]
# input = np.array(input)
# output = dummy(input)
#
# tf_fun = tf.py_func(dummy,[input],tf.float32)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     print(sess.run(tf_fun))
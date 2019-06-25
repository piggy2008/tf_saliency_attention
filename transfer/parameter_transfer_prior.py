import scipy.io as sio
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from matplotlib import pyplot as plt
from cell import ConvLSTMCell

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def astro_conv2d(x, w, hole=2):
    return tf.nn.atrous_conv2d(x, w, rate=hole, padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')



# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
parameter = sio.loadmat('../mat_parameter/fusionST_parameter_ms.mat')
parameter2 = sio.loadmat('../mat_parameter/DCL_parameter.mat')

for key in parameter2.keys():
    print(key)

################### R2 ###########################

conv1_1_r2_w = parameter['conv1_1_r2_w']
# conv1_1_r2_w = conv1_1_r2_w[:, :3, :, :]
conv1_1_r2_b = parameter['conv1_1_r2_b']

conv1_2_r2_w = parameter['conv1_2_r2_w']
conv1_2_r2_b = parameter['conv1_2_r2_b']

conv2_1_r2_w = parameter['conv2_1_r2_w']
conv2_1_r2_b = parameter['conv2_1_r2_b']

conv2_2_r2_w = parameter['conv2_2_r2_w']
conv2_2_r2_b = parameter['conv2_2_r2_b']

conv3_1_r2_w = parameter['conv3_1_r2_w']
conv3_1_r2_b = parameter['conv3_1_r2_b']

conv3_2_r2_w = parameter['conv3_2_r2_w']
conv3_2_r2_b = parameter['conv3_2_r2_b']

conv3_3_r2_w = parameter['conv3_3_r2_w']
conv3_3_r2_b = parameter['conv3_3_r2_b']

conv4_1_r2_w = parameter['conv4_1_r2_w']
conv4_1_r2_b = parameter['conv4_1_r2_b']

conv4_2_r2_w = parameter['conv4_2_r2_w']
conv4_2_r2_b = parameter['conv4_2_r2_b']

conv4_3_r2_w = parameter['conv4_3_r2_w']
conv4_3_r2_b = parameter['conv4_3_r2_b']

conv5_1_r2_w = parameter['conv5_1_r2_w']
conv5_1_r2_b = parameter['conv5_1_r2_b']

conv5_2_r2_w = parameter['conv5_2_r2_w']
conv5_2_r2_b = parameter['conv5_2_r2_b']

conv5_3_r2_w = parameter['conv5_3_r2_w']
conv5_3_r2_b = parameter['conv5_3_r2_b']

fc6_r2_w = parameter['fc6_r2_w']
fc6_r2_b = parameter['fc6_r2_b']

fc7_r2_w = parameter['fc7_r2_w']
fc7_r2_b = parameter['fc7_r2_b']

fc8_r2_w = parameter['fc8_saliency_r2_w']
fc8_r2_b = parameter['fc8_saliency_r2_b']

pool4_conv_r2_w = parameter['pool4_conv_r2_w']
pool4_conv_r2_b = parameter['pool4_conv_r2_b']

pool4_fc_r2_w = parameter['pool4_fc_r2_w']
pool4_fc_r2_b = parameter['pool4_fc_r2_b']

pool4_ms_saliency_r2_w = parameter['pool4_ms_saliency_r2_w']
pool4_ms_saliency_r2_b = parameter['pool4_ms_saliency_r2_b']

pool4_conv_r2_w = parameter['pool4_conv_r2_w']
pool4_conv_r2_b = parameter['pool4_conv_r2_b']

pool4_fc_r2_w = parameter['pool4_fc_r2_w']
pool4_fc_r2_b = parameter['pool4_fc_r2_b']

pool4_ms_saliency_r2_w = parameter['pool4_ms_saliency_r2_w']
pool4_ms_saliency_r2_b = parameter['pool4_ms_saliency_r2_b']

pool3_conv_r2_w = parameter2['pool3_conv_w']
pool3_conv_r2_b = parameter2['pool3_conv_b']

pool3_fc_r2_w = parameter2['pool3_fc_w']
pool3_fc_r2_b = parameter2['pool3_fc_b']

pool3_ms_saliency_r2_w = parameter2['pool3_ms_saliency_w']
pool3_ms_saliency_r2_b = parameter2['pool3_ms_saliency_b']

pool2_conv_r2_w = parameter2['pool2_conv_w']
pool2_conv_r2_b = parameter2['pool2_conv_b']

pool2_fc_r2_w = parameter2['pool2_fc_w']
pool2_fc_r2_b = parameter2['pool2_fc_b']

pool2_ms_saliency_r2_w = parameter2['pool2_ms_saliency_w']
pool2_ms_saliency_r2_b = parameter2['pool2_ms_saliency_b']

pool1_conv_r2_w = parameter2['pool1_conv_w']
pool1_conv_r2_b = parameter2['pool1_conv_b']

pool1_fc_r2_w = parameter2['pool1_fc_w']
pool1_fc_r2_b = parameter2['pool1_fc_b']

pool1_ms_saliency_r2_w = parameter2['pool1_ms_saliency_w']
pool1_ms_saliency_r2_b = parameter2['pool1_ms_saliency_b']

############# ST fusion #############

pool3_saliency_ST_w = parameter['pool4_saliency_ST_w']
pool3_saliency_ST_b = parameter['pool4_saliency_ST_b']

pool4_saliency_ST_w = parameter['pool4_saliency_ST_w']
pool4_saliency_ST_b = parameter['pool4_saliency_ST_b']

fc8_saliency_ST_w = parameter['fc8_saliency_ST_w']
fc8_saliency_ST_b = parameter['fc8_saliency_ST_b']

########### tensorflow structure ##############

########### R2 weight ##############
conv1_1_r2_w = tf.Variable(np.transpose(conv1_1_r2_w, [2, 3, 1, 0]), name='conv1_1_r2_w')
conv1_2_r2_w = tf.Variable(np.transpose(conv1_2_r2_w, [2, 3, 1, 0]), name='conv1_2_r2_w')
conv2_1_r2_w = tf.Variable(np.transpose(conv2_1_r2_w, [2, 3, 1, 0]), name='conv2_1_r2_w')
conv2_2_r2_w = tf.Variable(np.transpose(conv2_2_r2_w, [2, 3, 1, 0]), name='conv2_2_r2_w')
conv3_1_r2_w = tf.Variable(np.transpose(conv3_1_r2_w, [2, 3, 1, 0]), name='conv3_1_r2_w')
conv3_2_r2_w = tf.Variable(np.transpose(conv3_2_r2_w, [2, 3, 1, 0]), name='conv3_2_r2_w')
conv3_3_r2_w = tf.Variable(np.transpose(conv3_3_r2_w, [2, 3, 1, 0]), name='conv3_3_r2_w')
conv4_1_r2_w = tf.Variable(np.transpose(conv4_1_r2_w, [2, 3, 1, 0]), name='conv4_1_r2_w')
conv4_2_r2_w = tf.Variable(np.transpose(conv4_2_r2_w, [2, 3, 1, 0]), name='conv4_2_r2_w')
conv4_3_r2_w = tf.Variable(np.transpose(conv4_3_r2_w, [2, 3, 1, 0]), name='conv4_3_r2_w')

conv5_1_r2_w = tf.Variable(np.transpose(conv5_1_r2_w, [2, 3, 1, 0]), name='conv5_1_r2_w')
conv5_2_r2_w = tf.Variable(np.transpose(conv5_2_r2_w, [2, 3, 1, 0]), name='conv5_2_r2_w')
conv5_3_r2_w = tf.Variable(np.transpose(conv5_3_r2_w, [2, 3, 1, 0]), name='conv5_3_r2_w')

fc6_r2_w = tf.Variable(np.transpose(fc6_r2_w, [2, 3, 1, 0]), name='fc6_r2_w')
fc7_r2_w = tf.Variable(np.transpose(fc7_r2_w, [2, 3, 1, 0]), name='fc7_r2_w')
fc8_r2_w = tf.Variable(np.transpose(fc8_r2_w, [2, 3, 1, 0]), name='fc8_r2_w')

pool5_conv_r2_w = tf.Variable(np.transpose(pool4_conv_r2_w, [2, 3, 1, 0]), name='pool5_conv_r2_w')
pool5_fc_r2_w = tf.Variable(np.transpose(pool4_fc_r2_w, [2, 3, 1, 0]), name='pool5_fc_r2_w')
pool5_ms_saliency_r2_w = tf.Variable(np.transpose(pool4_ms_saliency_r2_w, [2, 3, 1, 0]), name='pool5_ms_saliency_r2_w')

pool4_conv_r2_w = tf.Variable(np.transpose(pool4_conv_r2_w, [2, 3, 1, 0]), name='pool4_conv_r2_w')
pool4_fc_r2_w = tf.Variable(np.transpose(pool4_fc_r2_w, [2, 3, 1, 0]), name='pool4_fc_r2_w')
pool4_ms_saliency_r2_w = tf.Variable(np.transpose(pool4_ms_saliency_r2_w, [2, 3, 1, 0]), name='pool4_ms_saliency_r2_w')

pool3_conv_r2_w = tf.Variable(np.transpose(pool3_conv_r2_w, [2, 3, 1, 0]), name='pool3_conv_r2_w')
pool3_fc_r2_w = tf.Variable(np.transpose(pool3_fc_r2_w, [2, 3, 1, 0]), name='pool3_fc_r2_w')
pool3_ms_saliency_r2_w = tf.Variable(np.transpose(pool3_ms_saliency_r2_w, [2, 3, 1, 0]), name='pool3_ms_saliency_r2_w')

pool2_conv_r2_w = tf.Variable(np.transpose(pool2_conv_r2_w, [2, 3, 1, 0]), name='pool2_conv_r2_w')
pool2_fc_r2_w = tf.Variable(np.transpose(pool2_fc_r2_w, [2, 3, 1, 0]), name='pool2_fc_r2_w')
pool2_ms_saliency_r2_w = tf.Variable(np.transpose(pool2_ms_saliency_r2_w, [2, 3, 1, 0]), name='pool2_ms_saliency_r2_w')

pool1_conv_r2_w = tf.Variable(np.transpose(pool1_conv_r2_w, [2, 3, 1, 0]), name='pool1_conv_r2_w')
pool1_fc_r2_w = tf.Variable(np.transpose(pool1_fc_r2_w, [2, 3, 1, 0]), name='pool1_fc_r2_w')
pool1_ms_saliency_r2_w = tf.Variable(np.transpose(pool1_ms_saliency_r2_w, [2, 3, 1, 0]), name='pool1_ms_saliency_r2_w')

############ R2 bias ############
conv1_1_r2_b = tf.Variable(conv1_1_r2_b, name='conv1_1_r2_b')
conv1_2_r2_b = tf.Variable(conv1_2_r2_b, name='conv1_2_r2_b')
conv2_1_r2_b = tf.Variable(conv2_1_r2_b, name='conv2_1_r2_b')
conv2_2_r2_b = tf.Variable(conv2_2_r2_b, name='conv2_2_r2_b')
conv3_1_r2_b = tf.Variable(conv3_1_r2_b, name='conv3_1_r2_b')
conv3_2_r2_b = tf.Variable(conv3_2_r2_b, name='conv3_2_r2_b')
conv3_3_r2_b = tf.Variable(conv3_3_r2_b, name='conv3_3_r2_b')
conv4_1_r2_b = tf.Variable(conv4_1_r2_b, name='conv4_1_r2_b')
conv4_2_r2_b = tf.Variable(conv4_2_r2_b, name='conv4_2_r2_b')
conv4_3_r2_b = tf.Variable(conv4_3_r2_b, name='conv4_3_r2_b')
conv5_1_r2_b = tf.Variable(conv5_1_r2_b, name='conv5_1_r2_b')
conv5_2_r2_b = tf.Variable(conv5_2_r2_b, name='conv5_2_r2_b')
conv5_3_r2_b = tf.Variable(conv5_3_r2_b, name='conv5_3_r2_b')

fc6_r2_b = tf.Variable(fc6_r2_b, name='fc6_r2_b')
fc7_r2_b = tf.Variable(fc7_r2_b, name='fc7_r2_b')
fc8_r2_b = tf.Variable(fc8_r2_b, name='fc8_r2_b')

pool5_conv_r2_b = tf.Variable(pool4_conv_r2_b, name='pool5_conv_r2_b')
pool5_fc_r2_b = tf.Variable(pool4_fc_r2_b, name='pool5_fc_r2_b')
pool5_ms_saliency_r2_b = tf.Variable(pool4_ms_saliency_r2_b, name='pool5_ms_saliency_r2_b')

pool4_conv_r2_b = tf.Variable(pool4_conv_r2_b, name='pool4_conv_r2_b')
pool4_fc_r2_b = tf.Variable(pool4_fc_r2_b, name='pool4_fc_r2_b')
pool4_ms_saliency_r2_b = tf.Variable(pool4_ms_saliency_r2_b, name='pool4_ms_saliency_r2_b')

pool3_conv_r2_b = tf.Variable(pool3_conv_r2_b, name='pool3_conv_r2_b')
pool3_fc_r2_b = tf.Variable(pool3_fc_r2_b, name='pool3_fc_r2_b')
pool3_ms_saliency_r2_b = tf.Variable(pool3_ms_saliency_r2_b, name='pool3_ms_saliency_r2_b')

pool2_conv_r2_b = tf.Variable(pool2_conv_r2_b, name='pool2_conv_r2_b')
pool2_fc_r2_b = tf.Variable(pool2_fc_r2_b, name='pool2_fc_r2_b')
pool2_ms_saliency_r2_b = tf.Variable(pool2_ms_saliency_r2_b, name='pool2_ms_saliency_r2_b')

pool1_conv_r2_b = tf.Variable(pool1_conv_r2_b, name='pool1_conv_r2_b')
pool1_fc_r2_b = tf.Variable(pool1_fc_r2_b, name='pool1_fc_r2_b')
pool1_ms_saliency_r2_b = tf.Variable(pool1_ms_saliency_r2_b, name='pool1_ms_saliency_r2_b')

########## ST fusion ##########
# pool3_saliency_ST_w = tf.Variable(np.transpose(pool3_saliency_ST_w, [2, 3, 1, 0]), name='pool3_saliency_ST_w')
# pool4_saliency_ST_w = tf.Variable(np.transpose(pool4_saliency_ST_w, [2, 3, 1, 0]), name='pool4_saliency_ST_w')
# fc8_saliency_ST_w = tf.Variable(np.transpose(fc8_saliency_ST_w, [2, 3, 1, 0]), name='fc8_saliency_ST_w')
#
# pool3_saliency_ST_b = tf.Variable(pool3_saliency_ST_b, name='pool3_saliency_ST_b')
# pool4_saliency_ST_b = tf.Variable(pool4_saliency_ST_b, name='pool4_saliency_ST_b')
# fc8_saliency_ST_b = tf.Variable(fc8_saliency_ST_b, name='fc8_saliency_ST_b')

size = 512

# x = tf.placeholder(tf.float32, [4, size, size, 3])
x_prior = tf.placeholder(tf.float32, [4, size, size, 4])
# input: raw flow map
# x_prior = tf.placeholder(tf.float32, [4, size, size, 3])

####### R1 compute ###########

####### R2 compute ###########
conv1_1_r2 = tf.nn.relu(conv2d(x_prior, conv1_1_r2_w) + conv1_1_r2_b)
conv1_2_r2 = tf.nn.relu(conv2d(conv1_1_r2, conv1_2_r2_w) + conv1_2_r2_b)
pool1_r2 = tf.nn.max_pool(conv1_2_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
conv2_1_r2 = tf.nn.relu(conv2d(pool1_r2, conv2_1_r2_w) + conv2_1_r2_b)
conv2_2_r2 = tf.nn.relu(conv2d(conv2_1_r2, conv2_2_r2_w) + conv2_2_r2_b)
pool2_r2 = tf.nn.max_pool(conv2_2_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
conv3_1_r2 = tf.nn.relu(conv2d(pool2_r2, conv3_1_r2_w) + conv3_1_r2_b)
conv3_2_r2 = tf.nn.relu(conv2d(conv3_1_r2, conv3_2_r2_w) + conv3_2_r2_b)
conv3_3_r2 = tf.nn.relu(conv2d(conv3_2_r2, conv3_3_r2_w) + conv3_3_r2_b)
pool3_r2 = tf.nn.max_pool(conv3_3_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
conv4_1_r2 = tf.nn.relu(conv2d(pool3_r2, conv4_1_r2_w) + conv4_1_r2_b)
conv4_2_r2 = tf.nn.relu(conv2d(conv4_1_r2, conv4_2_r2_w) + conv4_2_r2_b)
conv4_3_r2 = tf.nn.relu(conv2d(conv4_2_r2, conv4_3_r2_w) + conv4_3_r2_b)
pool4_r2 = tf.nn.max_pool(conv4_3_r2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

conv5_1_r2 = tf.nn.relu(astro_conv2d(pool4_r2, conv5_1_r2_w, hole=2) + conv5_1_r2_b)
conv5_2_r2 = tf.nn.relu(astro_conv2d(conv5_1_r2, conv5_2_r2_w, hole=2) + conv5_2_r2_b)
conv5_3_r2 = tf.nn.relu(astro_conv2d(conv5_2_r2, conv5_3_r2_w, hole=2) + conv5_3_r2_b)
pool5_r2 = tf.nn.max_pool(conv5_3_r2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

fc6_r2 = tf.nn.relu(astro_conv2d(pool5_r2, fc6_r2_w, hole=4) + fc6_r2_b)
fc6_r2_dropout = tf.nn.dropout(fc6_r2, 0.5)

fc7_r2 = tf.nn.relu(astro_conv2d(fc6_r2_dropout, fc7_r2_w, hole=4) + fc7_r2_b)
fc7_r2_dropout = tf.nn.dropout(fc7_r2, 0.5)

fc8_r2 = conv2d(fc7_r2_dropout, fc8_r2_w) + fc8_r2_b

pool5_conv_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool5_r2, pool5_conv_r2_w) + pool5_conv_r2_b), 0.5)
pool5_fc_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool5_conv_r2, pool5_fc_r2_w) + pool5_fc_r2_b), 0.5)
pool5_ms_saliency_r2 = conv2d(pool5_fc_r2, pool5_ms_saliency_r2_w) + pool5_ms_saliency_r2_b

pool4_conv_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool4_r2, pool4_conv_r2_w) + pool4_conv_r2_b), 0.5)
pool4_fc_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool4_conv_r2, pool4_fc_r2_w) + pool4_fc_r2_b), 0.5)
pool4_ms_saliency_r2 = conv2d(pool4_fc_r2, pool4_ms_saliency_r2_w) + pool4_ms_saliency_r2_b

pool3_conv_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool3_r2, pool3_conv_r2_w) + pool3_conv_r2_b), 0.5)
pool3_fc_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool3_conv_r2, pool3_fc_r2_w) + pool3_fc_r2_b), 0.5)
pool3_ms_saliency_r2 = conv2d(pool3_fc_r2, pool3_ms_saliency_r2_w) + pool3_ms_saliency_r2_b

pool2_conv_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool2_r2, pool2_conv_r2_w) + pool2_conv_r2_b), 0.5)
pool2_fc_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool2_conv_r2, pool3_fc_r2_w) + pool2_fc_r2_b), 0.5)
pool2_ms_saliency_r2 = conv2d(pool2_fc_r2, pool2_ms_saliency_r2_w) + pool2_ms_saliency_r2_b

pool1_conv_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool1_r2, pool1_conv_r2_w) + pool1_conv_r2_b), 0.5)
pool1_fc_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool1_conv_r2, pool1_fc_r2_w) + pool1_fc_r2_b), 0.5)
pool1_ms_saliency_r2 = conv2d(pool1_fc_r2, pool1_ms_saliency_r2_w) + pool1_ms_saliency_r2_b

up_pool4_r2 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [128, 128])
up_pool3_r2 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [128, 128])
up_fc8_r2 = tf.image.resize_bilinear(fc8_r2, [128, 128])
final_saliency_r2 = tf.add(up_pool3_r2, up_pool4_r2)
final_saliency_r2 = tf.sigmoid(tf.add(final_saliency_r2, up_fc8_r2))

########## DSS structure ##########

# fc8
scale6_score = tf.image.resize_bilinear(fc8_r2, [size, size])

# fc8 + pool5
scale5_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 2, 1]), dtype=tf.float32, name='scale5_w')
scale5_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32, name='scale5_b')
scale5_score = conv2d(tf.concat([fc8_r2, pool5_ms_saliency_r2], axis=3), scale5_w) + scale5_b
scale5_score = tf.image.resize_bilinear(scale5_score, [size, size])

# fc8 + pool5 + pool4
scale4_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 3, 1]), dtype=tf.float32, name='scale4_w')
scale4_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32, name='scale4_b')
scale4_score = conv2d(tf.concat([fc8_r2, pool5_ms_saliency_r2, pool4_ms_saliency_r2], axis=3), scale4_w) + scale4_b
scale4_score = tf.image.resize_bilinear(scale4_score, [size, size])

# fc8 + pool5 + pool4 + pool3
scale3_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 4, 1]), dtype=tf.float32, name='scale3_w')
scale3_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32, name='scale3_b')

scale3_score = conv2d(tf.concat([fc8_r2, pool5_ms_saliency_r2, pool4_ms_saliency_r2, pool3_ms_saliency_r2], axis=3), scale3_w) + scale3_b
scale3_score = tf.image.resize_bilinear(scale3_score, [size, size])

# fc8 + pool5 + pool4 + pool3 + pool2
scale2_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 5, 1]), dtype=tf.float32, name='scale2_w')
scale2_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32, name='scale2_b')

pool2_size = pool2_ms_saliency_r2.get_shape().as_list()
up2_fc8 = tf.image.resize_bilinear(fc8_r2, [pool2_size[1], pool2_size[2]])
up2_pool5 = tf.image.resize_bilinear(pool5_ms_saliency_r2, [pool2_size[1], pool2_size[2]])
up2_pool4 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [pool2_size[1], pool2_size[2]])
up2_pool3 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [pool2_size[1], pool2_size[2]])
scale2_score = conv2d(tf.concat([up2_fc8, up2_pool5, up2_pool4, up2_pool3, pool2_ms_saliency_r2], axis=3), scale2_w) + scale2_b
scale2_score = tf.image.resize_bilinear(scale2_score, [size, size])

# fc8 + pool5 + pool4 + pool3 + pool2 + pool1
scale1_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 6, 1]), dtype=tf.float32, name='scale1_w')
scale1_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32, name='scale1_b')

pool1_size = pool1_ms_saliency_r2.get_shape().as_list()
up1_fc8 = tf.image.resize_bilinear(fc8_r2, [pool1_size[1], pool1_size[2]])
up1_pool5 = tf.image.resize_bilinear(pool5_ms_saliency_r2, [pool1_size[1], pool1_size[2]])
up1_pool4 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [pool1_size[1], pool1_size[2]])
up1_pool3 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [pool1_size[1], pool1_size[2]])
up1_pool2 = tf.image.resize_bilinear(pool2_ms_saliency_r2, [pool1_size[1], pool1_size[2]])
scale1_score = conv2d(tf.concat([up1_fc8, up1_pool5, up1_pool4, up1_pool3, up1_pool2, pool1_ms_saliency_r2], axis=3), scale1_w) + scale1_b
scale1_score = tf.image.resize_bilinear(scale2_score, [size, size])

########## rnn fusion ############


# inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2], axis=3), 0)
# cell = ConvLSTMCell([512, 512], 1, [3, 3])
# outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, scope='rnn')
# rnn_output = tf.squeeze(outputs, axis=0)
#
# inputs2 = tf.expand_dims(tf.concat([up_fc8, up_fc8_r2], axis=3), 0)
# cell2 = ConvLSTMCell([512, 512], 1, [3, 3])
# outputs2, state2 = tf.nn.dynamic_rnn(cell2, inputs2, dtype=inputs.dtype, scope='rnn2')
# rnn_output2 = tf.squeeze(outputs2, axis=0)

inputs = tf.expand_dims(tf.concat([scale1_score, scale2_score, scale3_score, scale4_score, scale5_score, scale6_score], axis=3), 0)
cell = ConvLSTMCell([512, 512], 6, [3, 3])

# conv3D_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 4, 1]),
#                                       dtype=tf.float32, name='3D_conv_w')
# conv3D_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1, 1]), dtype=tf.float32, name='3D_conv_b')
# C3D_outputs = tf.nn.conv3d(inputs, conv3D_w, strides=[1, 1, 1, 1, 1], padding='SAME', name='C3D') + conv3D_b

# outputs_static, outputs_dynamic = rnn_cell(inputs, 'rnn')

outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, scope='rnn')
rnn_output = tf.squeeze(outputs, axis=0)


# C3D_output = tf.squeeze(C3D_outputs, axis=0)
# up_rnn_output = tf.image.resize_bilinear(rnn_output, [size, size])
# up_C3D_output = tf.image.resize_bilinear(C3D_output, [size, size])
# attention_static = tf.multiply(tf.add(up_pool4, up_fc8), outputs_static)
# attention_dynamic = tf.multiply(tf.add(up_pool4_r2, up_fc8_r2), outputs_dynamic)
########## ST fusion #############
# pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3)
# pool4_saliency_ST = conv2d(pool4_saliency_cancat, pool4_saliency_ST_w) + pool4_saliency_ST_b
#
# pool3_saliency_cancat = tf.concat([pool3_ms_saliency, pool3_ms_saliency_r2], 3)
# pool3_saliency_ST = conv2d(pool3_saliency_cancat, pool3_saliency_ST_w) + pool3_saliency_ST_b
#
# fc8_cancat = tf.concat([fc8, fc8_r2], 3)
# fc8_saliency_ST = conv2d(fc8_cancat, fc8_saliency_ST_w) + fc8_saliency_ST_b
#
# pool3_saliency_ST_resize = tf.image.resize_bilinear(pool3_saliency_ST, [128, 128])
# pool4_saliency_ST_resize = tf.image.resize_bilinear(pool4_saliency_ST, [128, 128])
# fc8_saliency_ST_resize = tf.image.resize_bilinear(fc8_saliency_ST, [128, 128])

# pool4_fc8_combine_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 3, 1]), dtype=tf.float32, name='pool4_fc8_w')
# pool4_fc8_combine_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32, name='pool4_fc8_b')
# pool4_fc8_concat = tf.concat([pool3_saliency_ST_resize, pool4_saliency_ST_resize, fc8_saliency_ST_resize], axis=3)
# pool4_fc8_combine = conv2d(pool4_fc8_concat, pool4_fc8_combine_w) + pool4_fc8_combine_b
# pool4_fc8_combine = tf.add(pool3_saliency_ST_resize, pool4_saliency_ST_resize)
#
# pool4_fc8_combine = tf.sigmoid(tf.add(pool4_fc8_combine, fc8_saliency_ST_resize))
#
# motion_cancat = tf.concat([pool4_fc8_combine, C3D_output, rnn_output], axis=3)
# attention_conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 256]), dtype=tf.float32, name='attention_conv1_w')
# attention_conv1_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 256]), dtype=tf.float32, name='attention_conv1_b')
#
# attention_conv2_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 3]), dtype=tf.float32, name='attention_conv2_w')
# attention_conv2_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 3]), dtype=tf.float32, name='attention_conv2_b')
# attention_first = tf.nn.dropout(tf.nn.relu(conv2d(motion_cancat, attention_conv1_w) + attention_conv1_b), 0.5)
# attention_second = tf.nn.softmax(conv2d(attention_first, attention_conv2_w) + attention_conv2_b)
# final_fusion = tf.multiply(motion_cancat, attention_second)

final_saliency_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 6, 1]), dtype=tf.float32, name='final_saliency_w')
final_saliency_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32, name='final_saliency_b')

final_fusion = conv2d(rnn_output, final_saliency_w) + final_saliency_b

final_saliency = tf.reduce_sum(final_fusion, axis=3, keep_dims=True)
# ave_num = tf.constant(3.0, dtype=tf.float32, shape=[4, 128, 128, 1])
# final_saliency = tf.div(final_saliency, ave_num)
# up_final_saliency = tf.image.resize_bilinear(final_saliency, [size, size])
saver = tf.train.Saver()
init = tf.initialize_all_variables()
from image_data_loader import ImageAndPriorData
image_dir = '/home/ty/data/davis/480p'
label_dir = '/home/ty/data/davis/GT'
prior_dir = '/home/ty/data/davis/davis_flow_prior'
davis_file = open('/home/ty/data/davis/davis_file.txt')
image_names = [line.strip() for line in davis_file]
# dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
dataset = ImageAndPriorData(image_dir, label_dir, prior_dir, None, None, None, image_names,
                            None, '.jpg', '.png', 550, 512, 4, horizontal_flip=False)
x_, y_ = dataset.next_batch()
with tf.Session() as sess:
    sess.run(init)

    in_ = x_[:, :, :, :3]
    in2_ = x_[:, :, :, :]
    print (np.shape(in_))
    feed_dict = {x_prior: in2_}
    result1 = sess.run([final_saliency], feed_dict)
    # print result
    # print np.shape(result)
    # plt.imshow(result[0,:,:,0])
    # plt.show()
    saver.save(sess, '../fusion_dss_structure_parameter/fusionST_LSTM_DSS.ckpt')

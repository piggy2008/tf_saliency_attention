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


def rnn_cell(input_tensor, name):
    inputs = tf.split(input_tensor, num_or_size_splits=input_tensor.get_shape()[0], axis=0)
    reuse = False

    rnn_conv_variable_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 4, 4]),
                                      dtype=tf.float32, name=name + 'rnn_conv_w')
    rnn_conv_variable_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 4]), dtype=tf.float32, name=name + 'rnn_conv_b')

    rnn_static_variable_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 4, 1]), dtype=tf.float32,
                                             name='static_rnn_output_w')
    rnn_static_variable_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32,
                                             name='static_rnn_output_b')

    rnn_dynamic_variable_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 4, 1]), dtype=tf.float32,
                                        name='dynamic_rnn_output_w')
    rnn_dynamic_variable_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32,
                                        name='dynamic_rnn_output_b')

    outputs = []
    outputs_static_sq = []
    outputs_dynamic_sq = []
    for i, input in enumerate(inputs):
        if i == 0:  # initialize the hidden state to be the zero vector
            hiddenState_prev = tf.zeros((1, input.get_shape()[1], input.get_shape()[2], input.get_shape()[3]))
        else:
            hiddenState_prev = outputs[i - 1]

        with tf.variable_scope(name + '_rnn', reuse=reuse):
            w = tf.get_variable('w', shape=[3, 3, input.get_shape()[3], input.get_shape()[3]],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('b', shape=[1, 1, 1, input.get_shape()[3]],
                                initializer=tf.truncated_normal_initializer())
            h_current = tf.nn.conv2d(hiddenState_prev, w, strides=[1, 1, 1, 1], padding='SAME') + b

            input_current = tf.nn.conv2d(input, rnn_conv_variable_w, strides=[1, 1, 1, 1],
                                         padding='SAME') + rnn_conv_variable_b
            hiddenState = tf.nn.relu(tf.add(h_current, input_current))

            output_static_temp = tf.nn.conv2d(hiddenState, rnn_static_variable_w, strides=[1, 1, 1, 1],
                                       padding='SAME') + rnn_static_variable_b

            output_dynamic_temp = tf.nn.conv2d(hiddenState, rnn_dynamic_variable_w, strides=[1, 1, 1, 1],
                                       padding='SAME') + rnn_dynamic_variable_b

        outputs.append(hiddenState)
        outputs_static_sq.append(tf.squeeze(output_static_temp, 0))
        outputs_dynamic_sq.append(tf.squeeze(output_dynamic_temp, 0))
        reuse = True

    return tf.stack(outputs_static_sq, axis=0), tf.stack(outputs_dynamic_sq, axis=0)
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
parameter = sio.loadmat('../mat_parameter/fusionST_parameter_ms.mat')

conv1_1_w = parameter['conv1_1_w']
conv1_1_b = parameter['conv1_1_b']

conv1_2_w = parameter['conv1_2_w']
conv1_2_b = parameter['conv1_2_b']

conv2_1_w = parameter['conv2_1_w']
conv2_1_b = parameter['conv2_1_b']

conv2_2_w = parameter['conv2_2_w']
conv2_2_b = parameter['conv2_2_b']

conv3_1_w = parameter['conv3_1_w']
conv3_1_b = parameter['conv3_1_b']

conv3_2_w = parameter['conv3_2_w']
conv3_2_b = parameter['conv3_2_b']

conv3_3_w = parameter['conv3_3_w']
conv3_3_b = parameter['conv3_3_b']

conv4_1_w = parameter['conv4_1_w']
conv4_1_b = parameter['conv4_1_b']

conv4_2_w = parameter['conv4_2_w']
conv4_2_b = parameter['conv4_2_b']

conv4_3_w = parameter['conv4_3_w']
conv4_3_b = parameter['conv4_3_b']

conv5_1_w = parameter['conv5_1_w']
conv5_1_b = parameter['conv5_1_b']

conv5_2_w = parameter['conv5_2_w']
conv5_2_b = parameter['conv5_2_b']

conv5_3_w = parameter['conv5_3_w']
conv5_3_b = parameter['conv5_3_b']

fc6_w = parameter['fc6_w']
fc6_b = parameter['fc6_b']

fc7_w = parameter['fc7_w']
fc7_b = parameter['fc7_b']

fc8_w = parameter['fc8_saliency_w']
fc8_b = parameter['fc8_saliency_b']

pool4_conv_w = parameter['pool4_conv_w']
pool4_conv_b = parameter['pool4_conv_b']

pool4_fc_w = parameter['pool4_fc_w']
pool4_fc_b = parameter['pool4_fc_b']

pool4_ms_saliency_w = parameter['pool4_ms_saliency_w']
pool4_ms_saliency_b = parameter['pool4_ms_saliency_b']

pool3_conv_w = parameter['pool3_conv_w']
pool3_conv_b = parameter['pool3_conv_b']

pool3_fc_w = parameter['pool3_fc_w']
pool3_fc_b = parameter['pool3_fc_b']

pool3_ms_saliency_w = parameter['pool3_ms_saliency_w']
pool3_ms_saliency_b = parameter['pool3_ms_saliency_b']

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

pool3_conv_r2_w = parameter['pool3_conv_r2_w']
pool3_conv_r2_b = parameter['pool3_conv_r2_b']

pool3_fc_r2_w = parameter['pool3_fc_r2_w']
pool3_fc_r2_b = parameter['pool3_fc_r2_b']

pool3_ms_saliency_r2_w = parameter['pool3_ms_saliency_r2_w']
pool3_ms_saliency_r2_b = parameter['pool3_ms_saliency_r2_b']

############# ST fusion #############

pool3_saliency_ST_w = parameter['pool4_saliency_ST_w']
pool3_saliency_ST_b = parameter['pool4_saliency_ST_b']

pool4_saliency_ST_w = parameter['pool4_saliency_ST_w']
pool4_saliency_ST_b = parameter['pool4_saliency_ST_b']

fc8_saliency_ST_w = parameter['fc8_saliency_ST_w']
fc8_saliency_ST_b = parameter['fc8_saliency_ST_b']

########### tensorflow structure ##############
########### R1 weight ##############
conv1_1_w = tf.Variable(np.transpose(conv1_1_w, [2, 3, 1, 0]), name='conv1_1_w')

conv1_2_w = tf.Variable(np.transpose(conv1_2_w, [2, 3, 1, 0]), name='conv1_2_w')
conv2_1_w = tf.Variable(np.transpose(conv2_1_w, [2, 3, 1, 0]), name='conv2_1_w')
conv2_2_w = tf.Variable(np.transpose(conv2_2_w, [2, 3, 1, 0]), name='conv2_2_w')
conv3_1_w = tf.Variable(np.transpose(conv3_1_w, [2, 3, 1, 0]), name='conv3_1_w')
conv3_2_w = tf.Variable(np.transpose(conv3_2_w, [2, 3, 1, 0]), name='conv3_2_w')
conv3_3_w = tf.Variable(np.transpose(conv3_3_w, [2, 3, 1, 0]), name='conv3_3_w')
conv4_1_w = tf.Variable(np.transpose(conv4_1_w, [2, 3, 1, 0]), name='conv4_1_w')
conv4_2_w = tf.Variable(np.transpose(conv4_2_w, [2, 3, 1, 0]), name='conv4_2_w')
conv4_3_w = tf.Variable(np.transpose(conv4_3_w, [2, 3, 1, 0]), name='conv4_3_w')

conv5_1_w = tf.Variable(np.transpose(conv5_1_w, [2, 3, 1, 0]), name='conv5_1_w')
conv5_2_w = tf.Variable(np.transpose(conv5_2_w, [2, 3, 1, 0]), name='conv5_2_w')
conv5_3_w = tf.Variable(np.transpose(conv5_3_w, [2, 3, 1, 0]), name='conv5_3_w')

fc6_w = tf.Variable(np.transpose(fc6_w, [2, 3, 1, 0]), name='fc6_w')
fc7_w = tf.Variable(np.transpose(fc7_w, [2, 3, 1, 0]), name='fc7_w')
fc8_w = tf.Variable(np.transpose(fc8_w, [2, 3, 1, 0]), name='fc8_w')

pool4_conv_w = tf.Variable(np.transpose(pool4_conv_w, [2, 3, 1, 0]), name='pool4_conv_w')
pool4_fc_w = tf.Variable(np.transpose(pool4_fc_w, [2, 3, 1, 0]), name='pool4_fc_w')
pool4_ms_saliency_w = tf.Variable(np.transpose(pool4_ms_saliency_w, [2, 3, 1, 0]), name='pool4_ms_saliency_w')

pool3_conv_w = tf.Variable(np.transpose(pool3_conv_w, [2, 3, 1, 0]), name='pool3_conv_w')
pool3_fc_w = tf.Variable(np.transpose(pool3_fc_w, [2, 3, 1, 0]), name='pool3_fc_w')
pool3_ms_saliency_w = tf.Variable(np.transpose(pool3_ms_saliency_w, [2, 3, 1, 0]), name='pool3_ms_saliency_w')

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

pool4_conv_r2_w = tf.Variable(np.transpose(pool4_conv_r2_w, [2, 3, 1, 0]), name='pool4_conv_r2_w')
pool4_fc_r2_w = tf.Variable(np.transpose(pool4_fc_r2_w, [2, 3, 1, 0]), name='pool4_fc_r2_w')
pool4_ms_saliency_r2_w = tf.Variable(np.transpose(pool4_ms_saliency_r2_w, [2, 3, 1, 0]), name='pool4_ms_saliency_r2_w')

pool3_conv_r2_w = tf.Variable(np.transpose(pool3_conv_r2_w, [2, 3, 1, 0]), name='pool3_conv_r2_w')
pool3_fc_r2_w = tf.Variable(np.transpose(pool3_fc_r2_w, [2, 3, 1, 0]), name='pool3_fc_r2_w')
pool3_ms_saliency_r2_w = tf.Variable(np.transpose(pool3_ms_saliency_r2_w, [2, 3, 1, 0]), name='pool3_ms_saliency_r2_w')
############ R1 bias ############
conv1_1_b = tf.Variable(conv1_1_b, name='conv1_1_b')
conv1_2_b = tf.Variable(conv1_2_b, name='conv1_2_b')
conv2_1_b = tf.Variable(conv2_1_b, name='conv2_1_b')
conv2_2_b = tf.Variable(conv2_2_b, name='conv2_2_b')
conv3_1_b = tf.Variable(conv3_1_b, name='conv3_1_b')
conv3_2_b = tf.Variable(conv3_2_b, name='conv3_2_b')
conv3_3_b = tf.Variable(conv3_3_b, name='conv3_3_b')
conv4_1_b = tf.Variable(conv4_1_b, name='conv4_1_b')
conv4_2_b = tf.Variable(conv4_2_b, name='conv4_2_b')
conv4_3_b = tf.Variable(conv4_3_b, name='conv4_3_b')
conv5_1_b = tf.Variable(conv5_1_b, name='conv5_1_b')
conv5_2_b = tf.Variable(conv5_2_b, name='conv5_2_b')
conv5_3_b = tf.Variable(conv5_3_b, name='conv5_3_b')

fc6_b = tf.Variable(fc6_b, name='fc6_b')
fc7_b = tf.Variable(fc7_b, name='fc7_b')
fc8_b = tf.Variable(fc8_b, name='fc8_b')

pool4_conv_b = tf.Variable(pool4_conv_b, name='pool4_conv_b')
pool4_fc_b = tf.Variable(pool4_fc_b, name='pool4_fc_b')
pool4_ms_saliency_b = tf.Variable(pool4_ms_saliency_b, name='pool4_ms_saliency_b')

pool3_conv_b = tf.Variable(pool3_conv_b, name='pool3_conv_b')
pool3_fc_b = tf.Variable(pool3_fc_b, name='pool3_fc_b')
pool3_ms_saliency_b = tf.Variable(pool3_ms_saliency_b, name='pool3_ms_saliency_b')

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

pool4_conv_r2_b = tf.Variable(pool4_conv_r2_b, name='pool4_conv_r2_b')
pool4_fc_r2_b = tf.Variable(pool4_fc_r2_b, name='pool4_fc_r2_b')
pool4_ms_saliency_r2_b = tf.Variable(pool4_ms_saliency_r2_b, name='pool4_ms_saliency_r2_b')

pool3_conv_r2_b = tf.Variable(pool3_conv_r2_b, name='pool3_conv_r2_b')
pool3_fc_r2_b = tf.Variable(pool3_fc_r2_b, name='pool3_fc_r2_b')
pool3_ms_saliency_r2_b = tf.Variable(pool3_ms_saliency_r2_b, name='pool3_ms_saliency_r2_b')

########## ST fusion ##########
pool3_saliency_ST_w = tf.Variable(np.transpose(pool3_saliency_ST_w, [2, 3, 1, 0]), name='pool3_saliency_ST_w')
pool4_saliency_ST_w = tf.Variable(np.transpose(pool4_saliency_ST_w, [2, 3, 1, 0]), name='pool4_saliency_ST_w')
fc8_saliency_ST_w = tf.Variable(np.transpose(fc8_saliency_ST_w, [2, 3, 1, 0]), name='fc8_saliency_ST_w')

pool3_saliency_ST_b = tf.Variable(pool3_saliency_ST_b, name='pool3_saliency_ST_b')
pool4_saliency_ST_b = tf.Variable(pool4_saliency_ST_b, name='pool4_saliency_ST_b')
fc8_saliency_ST_b = tf.Variable(fc8_saliency_ST_b, name='fc8_saliency_ST_b')

size = 512

x = tf.placeholder(tf.float32, [4, size, size, 3])
x_prior = tf.placeholder(tf.float32, [4, size, size, 4])
# input: raw flow map
# x_prior = tf.placeholder(tf.float32, [4, size, size, 3])

####### R1 compute ###########
conv1_1 = tf.nn.relu(conv2d(x, conv1_1_w) + conv1_1_b)
conv1_2 = tf.nn.relu(conv2d(conv1_1, conv1_2_w) + conv1_2_b)
pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
conv2_1 = tf.nn.relu(conv2d(pool1, conv2_1_w) + conv2_1_b)
conv2_2 = tf.nn.relu(conv2d(conv2_1, conv2_2_w) + conv2_2_b)
pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
conv3_1 = tf.nn.relu(conv2d(pool2, conv3_1_w) + conv3_1_b)
conv3_2 = tf.nn.relu(conv2d(conv3_1, conv3_2_w) + conv3_2_b)
conv3_3 = tf.nn.relu(conv2d(conv3_2, conv3_3_w) + conv3_3_b)
pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
conv4_1 = tf.nn.relu(conv2d(pool3, conv4_1_w) + conv4_1_b)
conv4_2 = tf.nn.relu(conv2d(conv4_1, conv4_2_w) + conv4_2_b)
conv4_3 = tf.nn.relu(conv2d(conv4_2, conv4_3_w) + conv4_3_b)
pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

conv5_1 = tf.nn.relu(astro_conv2d(pool4, conv5_1_w, hole=2) + conv5_1_b)
conv5_2 = tf.nn.relu(astro_conv2d(conv5_1, conv5_2_w, hole=2) + conv5_2_b)
conv5_3 = tf.nn.relu(astro_conv2d(conv5_2, conv5_3_w, hole=2) + conv5_3_b)
pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

fc6 = tf.nn.relu(astro_conv2d(pool5, fc6_w, hole=4) + fc6_b)
fc6_dropout = tf.nn.dropout(fc6, 0.5)

fc7 = tf.nn.relu(astro_conv2d(fc6_dropout, fc7_w, hole=4) + fc7_b)
fc7_dropout = tf.nn.dropout(fc7, 0.5)

fc8 = conv2d(fc7_dropout, fc8_w) + fc8_b

# rnn_output_fc8 = rnn_cell(fc8, 'fc8')
# fc8 = tf.add(rnn_output_fc8, fc8)

up_fc8 = tf.image.resize_bilinear(fc8, [128, 128])

pool4_conv = tf.nn.dropout(tf.nn.relu(conv2d(pool4, pool4_conv_w) + pool4_conv_b), 0.5)
pool4_fc = tf.nn.dropout(tf.nn.relu(conv2d(pool4_conv, pool4_fc_w) + pool4_fc_b), 0.5)
pool4_ms_saliency = conv2d(pool4_fc, pool4_ms_saliency_w) + pool4_ms_saliency_b


pool3_conv = tf.nn.dropout(tf.nn.relu(conv2d(pool3, pool3_conv_w) + pool3_conv_b), 0.5)
pool3_fc = tf.nn.dropout(tf.nn.relu(conv2d(pool3_conv, pool3_fc_w) + pool3_fc_b), 0.5)
pool3_ms_saliency = conv2d(pool3_fc, pool3_ms_saliency_w) + pool3_ms_saliency_b

# rnn_output_pool4 = rnn_cell(pool4_ms_saliency, 'pool4')
# pool4_ms_saliency = tf.add(rnn_output_pool4, pool4_ms_saliency)

up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [128, 128])
up_pool3 = tf.image.resize_bilinear(pool3_ms_saliency, [128, 128])
final_saliency_r1 = tf.add(up_pool3, up_pool4)
final_saliency_r1 = tf.sigmoid(tf.add(final_saliency_r1, up_fc8))

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


pool4_conv_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool4_r2, pool4_conv_r2_w) + pool4_conv_r2_b), 0.5)
pool4_fc_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool4_conv_r2, pool4_fc_r2_w) + pool4_fc_r2_b), 0.5)
pool4_ms_saliency_r2 = conv2d(pool4_fc_r2, pool4_ms_saliency_r2_w) + pool4_ms_saliency_r2_b

pool3_conv_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool3_r2, pool3_conv_r2_w) + pool3_conv_r2_b), 0.5)
pool3_fc_r2 = tf.nn.dropout(tf.nn.relu(conv2d(pool3_conv_r2, pool3_fc_r2_w) + pool3_fc_r2_b), 0.5)
pool3_ms_saliency_r2 = conv2d(pool3_fc_r2, pool3_ms_saliency_r2_w) + pool3_ms_saliency_r2_b

up_pool4_r2 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [128, 128])
up_pool3_r2 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [128, 128])
up_fc8_r2 = tf.image.resize_bilinear(fc8_r2, [128, 128])
final_saliency_r2 = tf.add(up_pool3_r2, up_pool4_r2)
final_saliency_r2 = tf.sigmoid(tf.add(final_saliency_r2, up_fc8_r2))


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

inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2, up_fc8, up_fc8_r2], axis=3), 0)
cell = ConvLSTMCell([128, 128], 1, [3, 3])

conv3D_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 4, 1]),
                                      dtype=tf.float32, name='3D_conv_w')
conv3D_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1, 1]), dtype=tf.float32, name='3D_conv_b')
C3D_outputs = tf.nn.conv3d(inputs, conv3D_w, strides=[1, 1, 1, 1, 1], padding='SAME', name='C3D') + conv3D_b

# outputs_static, outputs_dynamic = rnn_cell(inputs, 'rnn')

outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, scope='rnn')
rnn_output = tf.squeeze(outputs, axis=0)
C3D_output = tf.squeeze(C3D_outputs, axis=0)
up_rnn_output = tf.image.resize_bilinear(rnn_output, [size, size])
up_C3D_output = tf.image.resize_bilinear(C3D_output, [size, size])
# attention_static = tf.multiply(tf.add(up_pool4, up_fc8), outputs_static)
# attention_dynamic = tf.multiply(tf.add(up_pool4_r2, up_fc8_r2), outputs_dynamic)
########## ST fusion #############
pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3)
pool4_saliency_ST = conv2d(pool4_saliency_cancat, pool4_saliency_ST_w) + pool4_saliency_ST_b

pool3_saliency_cancat = tf.concat([pool3_ms_saliency, pool3_ms_saliency_r2], 3)
pool3_saliency_ST = conv2d(pool3_saliency_cancat, pool3_saliency_ST_w) + pool3_saliency_ST_b

fc8_cancat = tf.concat([fc8, fc8_r2], 3)
fc8_saliency_ST = conv2d(fc8_cancat, fc8_saliency_ST_w) + fc8_saliency_ST_b

pool3_saliency_ST_resize = tf.image.resize_bilinear(pool3_saliency_ST, [128, 128])
pool4_saliency_ST_resize = tf.image.resize_bilinear(pool4_saliency_ST, [128, 128])
fc8_saliency_ST_resize = tf.image.resize_bilinear(fc8_saliency_ST, [128, 128])

# pool4_fc8_combine_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 3, 1]), dtype=tf.float32, name='pool4_fc8_w')
# pool4_fc8_combine_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32, name='pool4_fc8_b')
# pool4_fc8_concat = tf.concat([pool3_saliency_ST_resize, pool4_saliency_ST_resize, fc8_saliency_ST_resize], axis=3)
# pool4_fc8_combine = conv2d(pool4_fc8_concat, pool4_fc8_combine_w) + pool4_fc8_combine_b
pool4_fc8_combine = tf.add(pool3_saliency_ST_resize, pool4_saliency_ST_resize)

pool4_fc8_combine = tf.sigmoid(tf.add(pool4_fc8_combine, fc8_saliency_ST_resize))

motion_cancat = tf.concat([pool4_fc8_combine, C3D_output, rnn_output], axis=3)
attention_conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 256]), dtype=tf.float32, name='attention_conv1_w')
attention_conv1_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 256]), dtype=tf.float32, name='attention_conv1_b')

attention_conv2_w = tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 3]), dtype=tf.float32, name='attention_conv2_w')
attention_conv2_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 3]), dtype=tf.float32, name='attention_conv2_b')
attention_first = tf.nn.dropout(tf.nn.relu(conv2d(motion_cancat, attention_conv1_w) + attention_conv1_b), 0.5)
attention_second = tf.nn.softmax(conv2d(attention_first, attention_conv2_w) + attention_conv2_b)
final_fusion = tf.multiply(motion_cancat, attention_second)

final_saliency = tf.reduce_sum(final_fusion, axis=3, keep_dims=True)
ave_num = tf.constant(3.0, dtype=tf.float32, shape=[4, 128, 128, 1])
final_saliency = tf.div(final_saliency, ave_num)
up_final_saliency = tf.image.resize_bilinear(final_saliency, [size, size])
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
    in2_ = x_[:, :, :, :3]
    print (np.shape(in_))
    feed_dict = {x: in_, x_prior: in2_}
    result1, result2, result = sess.run([final_saliency_r1, final_saliency_r2, final_saliency], feed_dict)
    # print result
    # print np.shape(result)
    # plt.imshow(result[0,:,:,0])
    # plt.show()
    saver.save(sess, '../fusion_C3D_ms_attention_parameter/fusionST_C3D_ms_attention_flow_tensorflow.ckpt')

import scipy.io as sio
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from matplotlib import pyplot as plt

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def astro_conv2d(x, w, hole=2):
    return tf.nn.atrous_conv2d(x, w, rate=hole, padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
parameter = sio.loadmat('../parameters/fusionST_parameter.mat')

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

################### R2 ###########################

conv1_1_r2_w = parameter['conv1_1_r2_w']
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

############# ST fusion #############

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

########## ST fusion ##########
pool4_saliency_ST_w = tf.Variable(np.transpose(pool4_saliency_ST_w, [2, 3, 1, 0]), name='pool4_saliency_ST_w')
fc8_saliency_ST_w = tf.Variable(np.transpose(fc8_saliency_ST_w, [2, 3, 1, 0]), name='fc8_saliency_ST_w')

pool4_saliency_ST_b = tf.Variable(pool4_saliency_ST_b, name='pool4_saliency_ST_b')
fc8_saliency_ST_b = tf.Variable(fc8_saliency_ST_b, name='fc8_saliency_ST_b')

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
x_prior = tf.placeholder(tf.float32, [None, 512, 512, 4])

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
up_fc8 = tf.image.resize_bilinear(fc8, [512, 512])

pool4_conv = tf.nn.dropout(tf.nn.relu(conv2d(pool4, pool4_conv_w) + pool4_conv_b), 0.5)
pool4_fc = tf.nn.dropout(tf.nn.relu(conv2d(pool4_conv, pool4_fc_w) + pool4_fc_b), 0.5)
pool4_ms_saliency = conv2d(pool4_fc, pool4_ms_saliency_w) + pool4_ms_saliency_b

up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [512, 512])
final_saliency_r1 = tf.sigmoid(tf.add(up_pool4, up_fc8))

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

up_pool4_r2 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [512, 512])
up_fc8_r2 = tf.image.resize_bilinear(fc8_r2, [512, 512])
final_saliency_r2 = tf.sigmoid(tf.add(up_pool4_r2, up_fc8_r2))

########## ST fusion #############
pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3)
pool4_saliency_ST = conv2d(pool4_saliency_cancat, pool4_saliency_ST_w) + pool4_saliency_ST_b

fc8_cancat = tf.concat([fc8, fc8_r2], 3)
fc8_saliency_ST = conv2d(fc8_cancat, fc8_saliency_ST_w) + fc8_saliency_ST_b

pool4_saliency_ST_resize = tf.image.resize_bilinear(pool4_saliency_ST, [512, 512])
fc8_saliency_ST_resize = tf.image.resize_bilinear(fc8_saliency_ST, [512, 512])

final_saliency = tf.sigmoid(tf.add(pool4_saliency_ST_resize, fc8_saliency_ST_resize))

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
                            None, '.jpg', '.png', 550, 512, 1, horizontal_flip=False)
x_, y_ = dataset.next_batch()
with tf.Session() as sess:
    sess.run(init)

    in_ = x_[:, :, :, :3]
    in2_ = x_[:, :, :, :]
    print np.shape(in_)
    feed_dict = {x: in_, x_prior: in2_}
    result1, result2, result = sess.run([final_saliency_r1, final_saliency_r2, final_saliency], feed_dict)
    # print result
    # print np.shape(result)
    # plt.imshow(result[0,:,:,0])
    # plt.show()
    saver.save(sess, '../fusion_parameter/fusionST_tensorflow.ckpt')

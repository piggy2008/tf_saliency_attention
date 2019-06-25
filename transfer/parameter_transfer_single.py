import scipy.io as sio
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from matplotlib import pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cell import ConvLSTMCell

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_2(x, w_shape, name):
    w = tf.Variable(tf.truncated_normal(shape=w_shape), dtype=tf.float32, name=name + '_w')
    b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, w_shape[-1]]), dtype=tf.float32, name=name + '_b')
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b

def astro_conv2d(x, w, hole=2):
    return tf.nn.atrous_conv2d(x, w, rate=hole, padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


def rnn_cell(input_tensor, name):
    inputs = tf.split(input_tensor, num_or_size_splits=input_tensor.get_shape()[0], axis=0)
    reuse = False

    outputs = []
    outputs_sq = []
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

            output_temp = tf.nn.conv2d(hiddenState, rnn_output_variable_w, strides=[1, 1, 1, 1],
                                       padding='SAME') + rnn_output_variable_b

        outputs.append(hiddenState)
        outputs_sq.append(tf.squeeze(output_temp, 0))
        reuse = True

    return tf.stack(outputs_sq, axis=0)

def rnn_cell_fc(input_tensor, name):
    inputs = tf.split(input_tensor, num_or_size_splits=input_tensor.get_shape()[0], axis=0)
    reuse = False

    outputs = []
    outputs_sq = []
    for i, input in enumerate(inputs):
        if i == 0:  # initialize the hidden state to be the zero vector
            hiddenState_prev = tf.zeros((1, input.get_shape()[1]))
        else:
            hiddenState_prev = outputs[i - 1]

        with tf.variable_scope(name + '_rnn', reuse=reuse):
            w = tf.get_variable('w', shape=[input.get_shape()[1], input.get_shape()[1]],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('b', shape=[input.get_shape()[1]],
                                initializer=tf.truncated_normal_initializer())
            h_current = tf.matmul(hiddenState_prev, w) + b
            input_current = tf.matmul(input, rnn_input_fc_w) + rnn_input_fc_b
            hiddenState = tf.nn.dropout(tf.nn.relu(tf.add(h_current, input_current)), 0.5)

            output_temp = tf.nn.dropout(tf.matmul(hiddenState, rnn_output_fc_w) + rnn_output_fc_b, 0.5)
            output_temp = tf.reshape(output_temp, [1, 64, 64, 1])

        outputs.append(hiddenState)
        outputs_sq.append(tf.squeeze(output_temp, 0))
        reuse = True

    return tf.stack(outputs_sq, axis=0)
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
parameter = sio.loadmat('../mat_parameter/DCL_parameter.mat')

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

pool2_conv_w = parameter['pool2_conv_w']
pool2_conv_b = parameter['pool2_conv_b']

pool2_fc_w = parameter['pool2_fc_w']
pool2_fc_b = parameter['pool2_fc_b']

pool2_ms_saliency_w = parameter['pool2_ms_saliency_w']
pool2_ms_saliency_b = parameter['pool2_ms_saliency_b']

pool1_conv_w = parameter['pool1_conv_w']
pool1_conv_b = parameter['pool1_conv_b']

pool1_fc_w = parameter['pool1_fc_w']
pool1_fc_b = parameter['pool1_fc_b']

pool1_ms_saliency_w = parameter['pool1_ms_saliency_w']
pool1_ms_saliency_b = parameter['pool1_ms_saliency_b']

############ RNN parameter ##########
# rnn_conv_variable_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 2, 2]), dtype=tf.float32,
#                                                name='pool4_rnn_conv_w')
# rnn_conv_variable_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 2]), dtype=tf.float32,
#                                        name='pool4_rnn_conv_b')
#
# rnn_output_variable_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 2, 1]), dtype=tf.float32,
#                                        name='pool4_rnn_output_w')
# rnn_output_variable_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32,
#                                        name='pool4_rnn_output_b')

rnn_input_fc_w = tf.Variable(tf.truncated_normal(shape=[64*64*2, 64*64*2], stddev=0.001), dtype=tf.float32,
                                               name='pool4_rnn_input_w')
rnn_input_fc_b = tf.Variable(tf.truncated_normal(shape=[64*64*2], stddev=0.001), dtype=tf.float32,
                                       name='pool4_rnn_input_b')

rnn_output_fc_w = tf.Variable(tf.truncated_normal(shape=[64*64*2, 64*64*1], stddev=0.001), dtype=tf.float32,
                                       name='pool4_rnn_output_w')
rnn_output_fc_b = tf.Variable(tf.truncated_normal(shape=[64*64*1], stddev=0.001), dtype=tf.float32,
                                       name='pool4_rnn_output_b')
# rnn_pool4_fc_w = tf.Variable(np.transpose(pool4_fc_w, [2, 3, 1, 0]), name='rnn_pool4_fc_w')
# rnn_pool4_saliency_w = tf.Variable(np.transpose(pool4_ms_saliency_w, [2, 3, 1, 0]), name='rnn_pool4_saliency_w')
# rnn_pool4_fc_b = tf.Variable(pool4_fc_b, name='rnn_pool4_fc_b')
# rnn_pool4_saliency_b = tf.Variable(pool4_ms_saliency_b, name='rnn_pool4_saliency_b')
################### R2 ###########################

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

pool2_conv_w = tf.Variable(np.transpose(pool2_conv_w, [2, 3, 1, 0]), name='pool2_conv_w')
pool2_fc_w = tf.Variable(np.transpose(pool2_fc_w, [2, 3, 1, 0]), name='pool2_fc_w')
pool2_ms_saliency_w = tf.Variable(np.transpose(pool2_ms_saliency_w, [2, 3, 1, 0]), name='pool2_ms_saliency_w')

pool1_conv_w = tf.Variable(np.transpose(pool1_conv_w, [2, 3, 1, 0]), name='pool1_conv_w')
pool1_fc_w = tf.Variable(np.transpose(pool1_fc_w, [2, 3, 1, 0]), name='pool1_fc_w')
pool1_ms_saliency_w = tf.Variable(np.transpose(pool1_ms_saliency_w, [2, 3, 1, 0]), name='pool1_ms_saliency_w')
########### R2 weight ##############

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

pool2_conv_b = tf.Variable(pool2_conv_b, name='pool2_conv_b')
pool2_fc_b = tf.Variable(pool2_fc_b, name='pool2_fc_b')
pool2_ms_saliency_b = tf.Variable(pool2_ms_saliency_b, name='pool2_ms_saliency_b')

pool1_conv_b = tf.Variable(pool1_conv_b, name='pool1_conv_b')
pool1_fc_b = tf.Variable(pool1_fc_b, name='pool1_fc_b')
pool1_ms_saliency_b = tf.Variable(pool1_ms_saliency_b, name='pool1_ms_saliency_b')
############ R2 bias ############



########## ST fusion ##########


x = tf.placeholder(tf.float32, [4, 512, 512, 3])
x_prior = tf.placeholder(tf.float32, [4, 512, 512, 4])

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

up_fc8 = tf.image.resize_bilinear(fc8, [512, 512])

pool4_conv = tf.nn.dropout(tf.nn.relu(conv2d(pool4, pool4_conv_w) + pool4_conv_b), 0.5)
pool4_fc = tf.nn.dropout(tf.nn.relu(conv2d(pool4_conv, pool4_fc_w) + pool4_fc_b), 0.5)
pool4_ms_saliency = conv2d(pool4_fc, pool4_ms_saliency_w) + pool4_ms_saliency_b
up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [512, 512])

pool3_conv = tf.nn.dropout(tf.nn.relu(conv2d(pool3, pool3_conv_w) + pool3_conv_b), 0.5)
pool3_fc = tf.nn.dropout(tf.nn.relu(conv2d(pool3_conv, pool3_fc_w) + pool3_fc_b), 0.5)
pool3_ms_saliency = conv2d(pool3_fc, pool3_ms_saliency_w) + pool3_ms_saliency_b
up_pool3 = tf.image.resize_bilinear(pool3_ms_saliency, [512, 512])

pool2_conv = tf.nn.dropout(tf.nn.relu(conv2d(pool2, pool2_conv_w) + pool2_conv_b), 0.5)
pool2_fc = tf.nn.dropout(tf.nn.relu(conv2d(pool2_conv, pool2_fc_w) + pool2_fc_b), 0.5)
pool2_ms_saliency = conv2d(pool2_fc, pool2_ms_saliency_w) + pool2_ms_saliency_b
up_pool2 = tf.image.resize_bilinear(pool2_ms_saliency, [512, 512])

pool1_conv = tf.nn.dropout(tf.nn.relu(conv2d(pool1, pool1_conv_w) + pool1_conv_b), 0.5)
pool1_fc = tf.nn.dropout(tf.nn.relu(conv2d(pool1_conv, pool1_fc_w) + pool1_fc_b), 0.5)
pool1_ms_saliency = conv2d(pool1_fc, pool1_ms_saliency_w) + pool1_ms_saliency_b
up_pool1 = tf.image.resize_bilinear(pool1_ms_saliency, [512, 512])

# rnn_output_pool4 = rnn_cell(pool4_ms_saliency, 'pool4')
# pool4_ms_saliency = tf.add(rnn_output_pool4, pool4_ms_saliency)
# rnn_pool4 = rnn_cell(pool4_conv, 'pool4')
# rnn_pool4_fc = tf.nn.relu(conv2d(rnn_pool4, rnn_pool4_fc_w) + rnn_pool4_fc_b)
# rnn_pool4_saliency = conv2d(rnn_pool4_fc, rnn_pool4_saliency_w) + rnn_pool4_saliency_b
# up_rnn_pool4 = tf.image.resize_bilinear(rnn_pool4_saliency, [512, 512])
# rnn_pool4_fc8 = rnn_cell(tf.concat([pool4_ms_saliency, fc8], axis=3), 'pool4_fc8')
# up_rnn = tf.image.resize_bilinear(rnn_pool4_fc8, [512, 512])

# rnn_pool4_fc8 = rnn_cell(tf.concat([up_pool4, up_fc8], axis=3), 'pool4_fc8')
# inputs_pool3 = tf.expand_dims(tf.concat([up_pool2, up_pool3], axis=3), 0)
# cell_pool3 = ConvLSTMCell([512, 512], 1, [3, 3])
# outputs_pool3, state = tf.nn.dynamic_rnn(cell_pool3, inputs_pool3, dtype=inputs_pool3.dtype, scope='pool3_rnn')
# rnn_output_pool3 = tf.squeeze(outputs_pool3, axis=0)
#
# inputs_pool4 = tf.expand_dims(tf.concat([up_pool3, up_pool4], axis=3), 0)
# cell_pool4 = ConvLSTMCell([512, 512], 1, [3, 3])
# outputs_pool4, state = tf.nn.dynamic_rnn(cell_pool4, inputs_pool4, dtype=inputs_pool4.dtype, scope='pool4_rnn')
# rnn_output_pool4 = tf.squeeze(outputs_pool4, axis=0)
# rnn_output_pos = tf.squeeze(outputs[0], axis=0)
# rnn_output_rev = tf.squeeze(outputs[1], axis=0)
# inputs = tf.reshape(tf.concat([pool4_ms_saliency, fc8], axis=3), [4, 64*64*2])
# rnn_output = rnn_cell_fc(inputs, 'pool4')
# up_rnn = tf.image.resize_bilinear(rnn_output, [512, 512])

final_saliency = tf.add(up_pool4, up_fc8)
final_saliency = tf.add(final_saliency, up_pool3)
final_saliency = tf.add(final_saliency, up_pool2)
final_saliency = tf.add(final_saliency, up_pool1)
# final_saliency = tf.add(final_saliency, rnn_output_pool3)
# final_saliency = tf.add(final_saliency, rnn_output_pool4)
final_saliency_r1 = tf.sigmoid(final_saliency)

####### R2 compute ###########


########## ST fusion #############


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
    feed_dict = {x: in_, x_prior: in2_}
    result1 = sess.run(final_saliency_r1, feed_dict)
    # print result
    # print np.shape(result)
    # plt.imshow(result[0,:,:,0])
    # plt.show()
    saver.save(sess, '../DCL_parameter/DCL_tensorflow_all_pool.ckpt')

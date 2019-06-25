import tensorflow as tf
import numpy as np
import math
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from image_data_loader import ImageData, ImageAndPriorData, ImageAndPriorSeqData
import os
import time
from utils import preprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cell import ConvLSTMCell

def get_conv_weights(weight_shape, sess):
    return math.sqrt(2 / (9.0 * 64)) * sess.run(tf.truncated_normal(weight_shape))

def mean_averge_error(pred, target):
    error = abs(np.squeeze(pred) - np.squeeze(target))
    mae = np.sum(error) / error.size
    return mae

def resize_image_prior(image, prior, max_shape=510):
    w, h = image.size
    if max(w, h) > max_shape:
        if w > h:
            image = image.resize([max_shape, int(float(max_shape) / w * h)])
            prior = prior.resize([max_shape, int(float(max_shape) / w * h)])
        else:
            image = image.resize([int(float(max_shape) / h * w), max_shape])
            prior = prior.resize([int(float(max_shape) / h * w), max_shape])

    return image, prior

class DCL(object):

    def __init__(self, sess, batch_size, lr=0.0001, ckpt_dir='./parameters'):
        self.sess = sess
        self.ckpt_dir = ckpt_dir
        self.lr = lr
        self.batch_size = batch_size

        # self.build_static_rnn()
        self.build_static()
        # self.build_ST()
        # self.build_ST_RNN()

    def conv2d(self, x, w_shape, name):
        w = tf.Variable(tf.truncated_normal(shape=w_shape), dtype=tf.float32, name=name + '_w')
        b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, w_shape[-1]]), dtype=tf.float32, name=name + '_b')
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b

    def astro_conv2d(self, x, w_shape, hole=2, name='astro_conv'):
        w = tf.Variable(tf.truncated_normal(shape=w_shape), dtype=tf.float32, name=name + '_w')
        b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, w_shape[-1]]), dtype=tf.float32, name=name + '_b')
        return tf.nn.atrous_conv2d(x, w, rate=hole, padding='SAME') + b

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    def rnn_cell(self, input_tensor, name):
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
                w = tf.get_variable('w', shape=[3, 3, input.get_shape()[3], input.get_shape()[3]], initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('b', shape=[1, 1, 1, input.get_shape()[3]], initializer=tf.truncated_normal_initializer())
                h_current = tf.nn.conv2d(hiddenState_prev, w, strides=[1, 1, 1, 1], padding='SAME') + b
                input_current = tf.nn.conv2d(input, self.rnn_conv_variable_w, strides=[1, 1, 1, 1], padding='SAME') + self.rnn_conv_variable_b
                hiddenState = tf.nn.relu(tf.add(h_current, input_current))

                output_temp = tf.nn.conv2d(hiddenState, self.rnn_output_variable_w, strides=[1, 1, 1, 1],
                             padding='SAME') + self.rnn_output_variable_b

            outputs.append(hiddenState)
            outputs_sq.append(tf.squeeze(output_temp, 0))
            reuse = True

        return tf.stack(outputs_sq, axis=0)

    def rnn_cell_fc(self, input_tensor, name):
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
                input_current = tf.matmul(input, self.rnn_input_fc_w) + self.rnn_input_fc_b
                hiddenState = tf.nn.dropout(tf.nn.relu(tf.add(h_current, input_current)), 0.9)

                output_temp = tf.nn.dropout(tf.matmul(hiddenState, self.rnn_output_fc_w) + self.rnn_output_fc_b, 0.9)
                output_temp = tf.reshape(output_temp, [1, 64, 64, 1])

            outputs.append(hiddenState)
            outputs_sq.append(tf.squeeze(output_temp, 0))
            reuse = True

        return tf.stack(outputs_sq, axis=0)


    def build_ST_RNN(self):
        self.X = tf.placeholder(tf.float32, [self.batch_size, 512, 512, 3], name='rgb_image')
        self.X_prior = tf.placeholder(tf.float32, [self.batch_size, 512, 512, 4], name='rgb_prior_image')
        self.Y = tf.placeholder(tf.float32, [self.batch_size, 512, 512, 1], name='gt')


        ############### R1 ###############
        conv1_1 = tf.nn.relu(self.conv2d(self.X, [3, 3, 3, 64], 'conv1_1'))
        conv1_2 = tf.nn.relu(self.conv2d(conv1_1, [3, 3, 64, 64], 'conv1_2'))
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        conv2_1 = tf.nn.relu(self.conv2d(pool1, [3, 3, 64, 128], 'conv2_1'))
        conv2_2 = tf.nn.relu(self.conv2d(conv2_1, [3, 3, 128, 128], 'conv2_2'))
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        conv3_1 = tf.nn.relu(self.conv2d(pool2, [3, 3, 128, 256], 'conv3_1'))
        conv3_2 = tf.nn.relu(self.conv2d(conv3_1, [3, 3, 256, 256], 'conv3_2'))
        conv3_3 = tf.nn.relu(self.conv2d(conv3_2, [3, 3, 256, 256], 'conv3_3'))
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        conv4_1 = tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 512], 'conv4_1'))
        conv4_2 = tf.nn.relu(self.conv2d(conv4_1, [3, 3, 512, 512], 'conv4_2'))
        conv4_3 = tf.nn.relu(self.conv2d(conv4_2, [3, 3, 512, 512], 'conv4_3'))
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')

        conv5_1 = tf.nn.relu(self.astro_conv2d(pool4, [3, 3, 512, 512], hole=2, name='conv5_1'))
        conv5_2 = tf.nn.relu(self.astro_conv2d(conv5_1, [3, 3, 512, 512], hole=2, name='conv5_2'))
        conv5_3 = tf.nn.relu(self.astro_conv2d(conv5_2, [3, 3, 512, 512], hole=2, name='conv5_3'))
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        fc6 = tf.nn.relu(self.astro_conv2d(pool5, [4, 4, 512, 4096], hole=4, name='fc6'))
        fc6_dropout = tf.nn.dropout(fc6, 0.5)

        fc7 = tf.nn.relu(self.astro_conv2d(fc6_dropout, [1, 1, 4096, 4096], hole=4, name='fc7'))
        fc7_dropout = tf.nn.dropout(fc7, 0.5)

        fc8 = self.conv2d(fc7_dropout, [1, 1, 4096, 1], 'fc8')
        rnn_output_fc8 = self.rnn_cell(fc8, 'fc8')
        fc8 = tf.add(rnn_output_fc8, fc8)

        up_fc8 = tf.image.resize_bilinear(fc8, [512, 512])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')

        rnn_output_pool4 = self.rnn_cell(pool4_ms_saliency, 'pool4')
        pool4_ms_saliency = tf.add(rnn_output_pool4, pool4_ms_saliency)

        up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [512, 512])
        final_saliency_r1 = tf.add(up_pool4, up_fc8)
        ############### R2 ###############
        conv1_1_r2 = tf.nn.relu(self.conv2d(self.X_prior, [3, 3, 4, 64], 'conv1_1_r2'))
        conv1_2_r2 = tf.nn.relu(self.conv2d(conv1_1_r2, [3, 3, 64, 64], 'conv1_2_r2'))
        pool1_r2 = tf.nn.max_pool(conv1_2_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_r2')
        conv2_1_r2 = tf.nn.relu(self.conv2d(pool1_r2, [3, 3, 64, 128], 'conv2_1_r2'))
        conv2_2_r2 = tf.nn.relu(self.conv2d(conv2_1_r2, [3, 3, 128, 128], 'conv2_2_r2'))
        pool2_r2 = tf.nn.max_pool(conv2_2_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_r2')
        conv3_1_r2 = tf.nn.relu(self.conv2d(pool2_r2, [3, 3, 128, 256], 'conv3_1_r2'))
        conv3_2_r2 = tf.nn.relu(self.conv2d(conv3_1_r2, [3, 3, 256, 256], 'conv3_2_r2'))
        conv3_3_r2 = tf.nn.relu(self.conv2d(conv3_2_r2, [3, 3, 256, 256], 'conv3_3_r2'))
        pool3_r2 = tf.nn.max_pool(conv3_3_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_r2')
        conv4_1_r2 = tf.nn.relu(self.conv2d(pool3_r2, [3, 3, 256, 512], 'conv4_1_r2'))
        conv4_2_r2 = tf.nn.relu(self.conv2d(conv4_1_r2, [3, 3, 512, 512], 'conv4_2_r2'))
        conv4_3_r2 = tf.nn.relu(self.conv2d(conv4_2_r2, [3, 3, 512, 512], 'conv4_3_r2'))
        pool4_r2 = tf.nn.max_pool(conv4_3_r2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4_r2')

        conv5_1_r2 = tf.nn.relu(self.astro_conv2d(pool4_r2, [3, 3, 512, 512], hole=2, name='conv5_1_r2'))
        conv5_2_r2 = tf.nn.relu(self.astro_conv2d(conv5_1_r2, [3, 3, 512, 512], hole=2, name='conv5_2_r2'))
        conv5_3_r2 = tf.nn.relu(self.astro_conv2d(conv5_2_r2, [3, 3, 512, 512], hole=2, name='conv5_3_r2'))
        pool5_r2 = tf.nn.max_pool(conv5_3_r2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        fc6_r2 = tf.nn.relu(self.astro_conv2d(pool5_r2, [4, 4, 512, 4096], hole=4, name='fc6_r2'))
        fc6_dropout_r2 = tf.nn.dropout(fc6_r2, 0.5)

        fc7_r2 = tf.nn.relu(self.astro_conv2d(fc6_dropout_r2, [1, 1, 4096, 4096], hole=4, name='fc7_r2'))
        fc7_dropout_r2 = tf.nn.dropout(fc7_r2, 0.5)

        fc8_r2 = self.conv2d(fc7_dropout_r2, [1, 1, 4096, 1], 'fc8_r2')
        up_fc8_r2 = tf.image.resize_bilinear(fc8_r2, [512, 512])

        pool4_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_r2, [3, 3, 512, 128], 'pool4_conv_r2')), 0.5)
        pool4_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv_r2, [1, 1, 128, 128], 'pool4_fc_r2')), 0.5)
        pool4_ms_saliency_r2 = self.conv2d(pool4_fc_r2, [1, 1, 128, 1], 'pool4_ms_saliency_r2')
        up_pool4_r2 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [512, 512])
        final_saliency_r2 = tf.add(up_pool4_r2, up_fc8_r2)

        ########### ST fusion ############
        pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3, name='concat_pool4')
        pool4_saliency_ST = self.conv2d(pool4_saliency_cancat, [1, 1, 2, 1], 'pool4_saliency_ST')
        up_pool4_ST = tf.image.resize_bilinear(pool4_saliency_ST, [512, 512])

        fc8_concat = tf.concat([fc8, fc8_r2], 3, name='concat_fc8')
        fc8_saliency_ST = self.conv2d(fc8_concat, [1, 1, 2, 1], 'fc8_saliency_ST')
        up_fc8_ST = tf.image.resize_bilinear(fc8_saliency_ST, [512, 512])

        final_saliency = tf.add(up_pool4_ST, up_fc8_ST)
        self.final_saliency = tf.sigmoid(final_saliency)
        self.up_fc8 = up_fc8
        self.saver = tf.train.Saver()

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=self.Y),
                                   name='loss')
        tf.summary.scalar('entropy', self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
        trainable_var = tf.trainable_variables()
        grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        self.train_op = optimizer.apply_gradients(grads)

    def build_ST(self):
        self.X = tf.placeholder(tf.float32, [None, 512, 512, 3], name='rgb_image')
        self.X_prior = tf.placeholder(tf.float32, [None, 512, 512, 4], name='rgb_prior_image')
        self.Y = tf.placeholder(tf.float32, [None, 512, 512, 1], name='gt')

        ############### R1 ###############
        conv1_1 = tf.nn.relu(self.conv2d(self.X, [3, 3, 3, 64], 'conv1_1'))
        conv1_2 = tf.nn.relu(self.conv2d(conv1_1, [3, 3, 64, 64], 'conv1_2'))
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        conv2_1 = tf.nn.relu(self.conv2d(pool1, [3, 3, 64, 128], 'conv2_1'))
        conv2_2 = tf.nn.relu(self.conv2d(conv2_1, [3, 3, 128, 128], 'conv2_2'))
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        conv3_1 = tf.nn.relu(self.conv2d(pool2, [3, 3, 128, 256], 'conv3_1'))
        conv3_2 = tf.nn.relu(self.conv2d(conv3_1, [3, 3, 256, 256], 'conv3_2'))
        conv3_3 = tf.nn.relu(self.conv2d(conv3_2, [3, 3, 256, 256], 'conv3_3'))
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        conv4_1 = tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 512], 'conv4_1'))
        conv4_2 = tf.nn.relu(self.conv2d(conv4_1, [3, 3, 512, 512], 'conv4_2'))
        conv4_3 = tf.nn.relu(self.conv2d(conv4_2, [3, 3, 512, 512], 'conv4_3'))
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')

        conv5_1 = tf.nn.relu(self.astro_conv2d(pool4, [3, 3, 512, 512], hole=2, name='conv5_1'))
        conv5_2 = tf.nn.relu(self.astro_conv2d(conv5_1, [3, 3, 512, 512], hole=2, name='conv5_2'))
        conv5_3 = tf.nn.relu(self.astro_conv2d(conv5_2, [3, 3, 512, 512], hole=2, name='conv5_3'))
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        fc6 = tf.nn.relu(self.astro_conv2d(pool5, [4, 4, 512, 4096], hole=4, name='fc6'))
        fc6_dropout = tf.nn.dropout(fc6, 0.5)

        fc7 = tf.nn.relu(self.astro_conv2d(fc6_dropout, [1, 1, 4096, 4096], hole=4, name='fc7'))
        fc7_dropout = tf.nn.dropout(fc7, 0.5)

        fc8 = self.conv2d(fc7_dropout, [1, 1, 4096, 1], 'fc8')
        up_fc8 = tf.image.resize_bilinear(fc8, [512, 512])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')
        up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [512, 512])
        final_saliency_r1 = tf.add(up_pool4, up_fc8)

        ############### R2 ###############
        conv1_1_r2 = tf.nn.relu(self.conv2d(self.X_prior, [3, 3, 4, 64], 'conv1_1_r2'))
        conv1_2_r2 = tf.nn.relu(self.conv2d(conv1_1_r2, [3, 3, 64, 64], 'conv1_2_r2'))
        pool1_r2 = tf.nn.max_pool(conv1_2_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1_r2')
        conv2_1_r2 = tf.nn.relu(self.conv2d(pool1_r2, [3, 3, 64, 128], 'conv2_1_r2'))
        conv2_2_r2 = tf.nn.relu(self.conv2d(conv2_1_r2, [3, 3, 128, 128], 'conv2_2_r2'))
        pool2_r2 = tf.nn.max_pool(conv2_2_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2_r2')
        conv3_1_r2 = tf.nn.relu(self.conv2d(pool2_r2, [3, 3, 128, 256], 'conv3_1_r2'))
        conv3_2_r2 = tf.nn.relu(self.conv2d(conv3_1_r2, [3, 3, 256, 256], 'conv3_2_r2'))
        conv3_3_r2 = tf.nn.relu(self.conv2d(conv3_2_r2, [3, 3, 256, 256], 'conv3_3_r2'))
        pool3_r2 = tf.nn.max_pool(conv3_3_r2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3_r2')
        conv4_1_r2 = tf.nn.relu(self.conv2d(pool3_r2, [3, 3, 256, 512], 'conv4_1_r2'))
        conv4_2_r2 = tf.nn.relu(self.conv2d(conv4_1_r2, [3, 3, 512, 512], 'conv4_2_r2'))
        conv4_3_r2 = tf.nn.relu(self.conv2d(conv4_2_r2, [3, 3, 512, 512], 'conv4_3_r2'))
        pool4_r2 = tf.nn.max_pool(conv4_3_r2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4_r2')

        conv5_1_r2 = tf.nn.relu(self.astro_conv2d(pool4_r2, [3, 3, 512, 512], hole=2, name='conv5_1_r2'))
        conv5_2_r2 = tf.nn.relu(self.astro_conv2d(conv5_1_r2, [3, 3, 512, 512], hole=2, name='conv5_2_r2'))
        conv5_3_r2 = tf.nn.relu(self.astro_conv2d(conv5_2_r2, [3, 3, 512, 512], hole=2, name='conv5_3_r2'))
        pool5_r2 = tf.nn.max_pool(conv5_3_r2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        fc6_r2 = tf.nn.relu(self.astro_conv2d(pool5_r2, [4, 4, 512, 4096], hole=4, name='fc6_r2'))
        fc6_dropout_r2 = tf.nn.dropout(fc6_r2, 0.5)

        fc7_r2 = tf.nn.relu(self.astro_conv2d(fc6_dropout_r2, [1, 1, 4096, 4096], hole=4, name='fc7_r2'))
        fc7_dropout_r2 = tf.nn.dropout(fc7_r2, 0.5)

        fc8_r2 = self.conv2d(fc7_dropout_r2, [1, 1, 4096, 1], 'fc8_r2')
        up_fc8_r2 = tf.image.resize_bilinear(fc8_r2, [512, 512])

        pool4_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_r2, [3, 3, 512, 128], 'pool4_conv_r2')), 0.5)
        pool4_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv_r2, [1, 1, 128, 128], 'pool4_fc_r2')), 0.5)
        pool4_ms_saliency_r2 = self.conv2d(pool4_fc_r2, [1, 1, 128, 1], 'pool4_ms_saliency_r2')
        up_pool4_r2 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [512, 512])
        final_saliency_r2 = tf.add(up_pool4_r2, up_fc8_r2)

        ########### ST fusion ############
        pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3, name='concat_pool4')
        pool4_saliency_ST = self.conv2d(pool4_saliency_cancat, [1, 1, 2, 1], 'pool4_saliency_ST')
        up_pool4_ST = tf.image.resize_bilinear(pool4_saliency_ST, [512, 512])

        fc8_concat = tf.concat([fc8, fc8_r2], 3, name='concat_fc8')
        fc8_saliency_ST = self.conv2d(fc8_concat, [1, 1, 2, 1], 'fc8_saliency_ST')
        up_fc8_ST = tf.image.resize_bilinear(fc8_saliency_ST, [512, 512])


        final_saliency = tf.add(up_pool4_ST, up_fc8_ST)
        self.final_saliency = tf.sigmoid(final_saliency)
        self.up_fc8 = up_fc8
        self.saver = tf.train.Saver()


        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=self.Y), name='loss')
        tf.summary.scalar('entropy', self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
        trainable_var = tf.trainable_variables()
        grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        self.train_op = optimizer.apply_gradients(grads)


    def build_static(self):
        self.X = tf.placeholder(tf.float32, [None, 512, 512, 3], name='rgb_image')
        self.Y = tf.placeholder(tf.float32, [None, 512, 512, 1], name='gt')

        conv1_1 = tf.nn.relu(self.conv2d(self.X, [3, 3, 3, 64], 'conv1_1'))
        conv1_2 = tf.nn.relu(self.conv2d(conv1_1, [3, 3, 64, 64], 'conv1_2'))
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        conv2_1 = tf.nn.relu(self.conv2d(pool1, [3, 3, 64, 128], 'conv2_1'))
        conv2_2 = tf.nn.relu(self.conv2d(conv2_1, [3, 3, 128, 128], 'conv2_2'))
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        conv3_1 = tf.nn.relu(self.conv2d(pool2, [3, 3, 128, 256], 'conv3_1'))
        conv3_2 = tf.nn.relu(self.conv2d(conv3_1, [3, 3, 256, 256], 'conv3_2'))
        conv3_3 = tf.nn.relu(self.conv2d(conv3_2, [3, 3, 256, 256], 'conv3_3'))
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        conv4_1 = tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 512], 'conv4_1'))
        conv4_2 = tf.nn.relu(self.conv2d(conv4_1, [3, 3, 512, 512], 'conv4_2'))
        conv4_3 = tf.nn.relu(self.conv2d(conv4_2, [3, 3, 512, 512], 'conv4_3'))
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')

        conv5_1 = tf.nn.relu(self.astro_conv2d(pool4, [3, 3, 512, 512], hole=2, name='conv5_1'))
        conv5_2 = tf.nn.relu(self.astro_conv2d(conv5_1, [3, 3, 512, 512], hole=2, name='conv5_2'))
        conv5_3 = tf.nn.relu(self.astro_conv2d(conv5_2, [3, 3, 512, 512], hole=2, name='conv5_3'))
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        fc6 = tf.nn.relu(self.astro_conv2d(pool5, [4, 4, 512, 4096], hole=4, name='fc6'))
        fc6_dropout = tf.nn.dropout(fc6, 0.5)

        fc7 = tf.nn.relu(self.astro_conv2d(fc6_dropout, [1, 1, 4096, 4096], hole=4, name='fc7'))
        fc7_dropout = tf.nn.dropout(fc7, 0.5)

        fc8 = self.conv2d(fc7_dropout, [1, 1, 4096, 1], 'fc8')
        up_fc8 = tf.image.resize_bilinear(fc8, [512, 512])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')
        up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [512, 512])
        final_saliency = tf.add(up_pool4, up_fc8)

        self.final_saliency = tf.sigmoid(final_saliency)
        self.up_fc8 = up_fc8
        self.saver = tf.train.Saver()


        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=self.Y), name='loss')
        tf.summary.scalar('entropy', self.loss)

        optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
        trainable_var = tf.trainable_variables()
        grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        self.train_op = optimizer.apply_gradients(grads)

    def build_static_rnn(self):
        self.X = tf.placeholder(tf.float32, [self.batch_size, 512, 512, 3], name='rgb_image')
        self.Y = tf.placeholder(tf.float32, [self.batch_size, 512, 512, 1], name='gt')

        conv1_1 = tf.nn.relu(self.conv2d(self.X, [3, 3, 3, 64], 'conv1_1'))
        conv1_2 = tf.nn.relu(self.conv2d(conv1_1, [3, 3, 64, 64], 'conv1_2'))
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        conv2_1 = tf.nn.relu(self.conv2d(pool1, [3, 3, 64, 128], 'conv2_1'))
        conv2_2 = tf.nn.relu(self.conv2d(conv2_1, [3, 3, 128, 128], 'conv2_2'))
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        conv3_1 = tf.nn.relu(self.conv2d(pool2, [3, 3, 128, 256], 'conv3_1'))
        conv3_2 = tf.nn.relu(self.conv2d(conv3_1, [3, 3, 256, 256], 'conv3_2'))
        conv3_3 = tf.nn.relu(self.conv2d(conv3_2, [3, 3, 256, 256], 'conv3_3'))
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        conv4_1 = tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 512], 'conv4_1'))
        conv4_2 = tf.nn.relu(self.conv2d(conv4_1, [3, 3, 512, 512], 'conv4_2'))
        conv4_3 = tf.nn.relu(self.conv2d(conv4_2, [3, 3, 512, 512], 'conv4_3'))
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')

        conv5_1 = tf.nn.relu(self.astro_conv2d(pool4, [3, 3, 512, 512], hole=2, name='conv5_1'))
        conv5_2 = tf.nn.relu(self.astro_conv2d(conv5_1, [3, 3, 512, 512], hole=2, name='conv5_2'))
        conv5_3 = tf.nn.relu(self.astro_conv2d(conv5_2, [3, 3, 512, 512], hole=2, name='conv5_3'))
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        fc6 = tf.nn.relu(self.astro_conv2d(pool5, [4, 4, 512, 4096], hole=4, name='fc6'))
        fc6_dropout = tf.nn.dropout(fc6, 0.5)

        fc7 = tf.nn.relu(self.astro_conv2d(fc6_dropout, [1, 1, 4096, 4096], hole=4, name='fc7'))
        fc7_dropout = tf.nn.dropout(fc7, 0.5)

        fc8 = self.conv2d(fc7_dropout, [1, 1, 4096, 1], 'fc8')
        up_fc8 = tf.image.resize_bilinear(fc8, [512, 512])

        pool1_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool1, [3, 3, 64, 128], 'pool1_conv')), 0.5)
        pool1_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool1_conv, [1, 1, 128, 128], 'pool1_fc')), 0.5)
        pool1_ms_saliency = self.conv2d(pool1_fc, [1, 1, 128, 1], 'pool1_ms_saliency')
        up_pool1 = tf.image.resize_bilinear(pool1_ms_saliency, [512, 512])

        pool2_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool2, [3, 3, 128, 128], 'pool2_conv')), 0.5)
        pool2_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool2_conv, [1, 1, 128, 128], 'pool2_fc')), 0.5)
        pool2_ms_saliency = self.conv2d(pool2_fc, [1, 1, 128, 1], 'pool2_ms_saliency')
        up_pool2 = tf.image.resize_bilinear(pool2_ms_saliency, [512, 512])

        pool3_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 128], 'pool3_conv')), 0.5)
        pool3_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_conv, [1, 1, 128, 128], 'pool3_fc')), 0.5)
        pool3_ms_saliency = self.conv2d(pool3_fc, [1, 1, 128, 1], 'pool3_ms_saliency')
        up_pool3 = tf.image.resize_bilinear(pool3_ms_saliency, [512, 512])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')
        up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [512, 512])

        self.pool4_ms_saliency = pool4_ms_saliency

        ########### rnn cell parameter ##############
        # self.rnn_conv_variable_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 2, 2]), dtype=tf.float32,
        #                                        name='pool4_rnn_conv_w')
        # self.rnn_conv_variable_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 2]), dtype=tf.float32,
        #                                        name='pool4_rnn_conv_b')
        #
        # self.rnn_output_variable_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 2, 1]), dtype=tf.float32,
        #                                          name='pool4_rnn_output_w')
        # self.rnn_output_variable_b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1]), dtype=tf.float32,
        #                                          name='pool4_rnn_output_b')

        # self.rnn_input_fc_w = tf.Variable(tf.truncated_normal(shape=[64 * 64 * 2, 64 * 64 * 2], stddev=0.001),
        #                              dtype=tf.float32,
        #                              name='pool4_rnn_input_w')
        # self.rnn_input_fc_b = tf.Variable(tf.truncated_normal(shape=[64 * 64 * 2], stddev=0.001), dtype=tf.float32,
        #                              name='pool4_rnn_input_b')
        #
        # self.rnn_output_fc_w = tf.Variable(tf.truncated_normal(shape=[64 * 64 * 2, 64 * 64 * 1], stddev=0.001),
        #                               dtype=tf.float32,
        #                               name='pool4_rnn_output_w')
        # self.rnn_output_fc_b = tf.Variable(tf.truncated_normal(shape=[64 * 64 * 1], stddev=0.001), dtype=tf.float32,
        #                               name='pool4_rnn_output_b')

        # rnn_pool4 = self.rnn_cell(pool4_conv, 'pool4')
        # rnn_pool4_fc = tf.nn.relu(self.conv2d(rnn_pool4, [1, 1, 128, 128], 'rnn_pool4_fc'))
        # rnn_pool4_saliency = self.conv2d(rnn_pool4_fc, [1, 1, 128, 1], 'rnn_pool4_saliency')
        # up_rnn_pool4 = tf.image.resize_bilinear(rnn_pool4_saliency, [512, 512])
        # rnn_pool4_fc8 = self.rnn_cell(tf.concat([up_pool4, up_fc8], axis=3), 'pool4_fc8')
        # up_rnn = tf.image.resize_bilinear(rnn_pool4_fc8, [512, 512])
        # rnn_fc8 = self.rnn_cell(up_fc8, 'fc8')
        # up_rnn = tf.image.resize_bilinear(rnn_pool4_fc8, [512, 512])

        # inputs = tf.reshape(tf.concat([pool4_ms_saliency, fc8], axis=3), [4, 64 * 64 * 2])
        # rnn_output = self.rnn_cell_fc(inputs, 'pool4')
        # up_rnn = tf.image.resize_bilinear(rnn_output, [512, 512])

        # inputs = tf.expand_dims(tf.concat([up_pool4, up_fc8], axis=3), 0)
        # cell = ConvLSTMCell([512, 512], 1, [3, 3])
        # outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)
        # rnn_output_pos = tf.squeeze(outputs[0], axis=0)
        # rnn_output_rev = tf.squeeze(outputs[1], axis=0)

        # inputs_pool4 = tf.expand_dims(tf.concat([up_pool3, up_pool4], axis=3), 0)
        # cell_pool4 = ConvLSTMCell([512, 512], 1, [3, 3])
        # outputs_pool4, state = tf.nn.dynamic_rnn(cell_pool4, inputs_pool4, dtype=inputs_pool4.dtype, scope='pool3_rnn')
        # rnn_output_pool4 = tf.squeeze(outputs_pool4, axis=0)
        #
        # inputs_fc8 = tf.expand_dims(tf.concat([up_pool4, up_fc8], axis=3), 0)
        # cell_fc8 = ConvLSTMCell([512, 512], 1, [3, 3])
        # outputs_fc8, state = tf.nn.dynamic_rnn(cell_fc8, inputs_fc8, dtype=inputs_fc8.dtype, scope='pool4_rnn')
        # rnn_output_fc8 = tf.squeeze(outputs_fc8, axis=0)

        final_saliency = tf.add(up_pool4, up_fc8)
        final_saliency = tf.add(final_saliency, up_pool3)
        final_saliency = tf.add(final_saliency, up_pool2)
        final_saliency = tf.add(final_saliency, up_pool1)
        # final_saliency = tf.add(final_saliency, rnn_output_pool4)
        # final_saliency = tf.add(final_saliency, rnn_output_fc8)
        self.final_saliency = tf.sigmoid(final_saliency)

        self.rnn_output = final_saliency
        self.saver = tf.train.Saver()


        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=self.Y), name='loss')
        tf.summary.scalar('entropy', self.loss)

        trainable_layer_len = 28

        optimizer1 = tf.train.AdamOptimizer(self.lr, name='optimizer1')
        optimizer2 = tf.train.AdamOptimizer(self.lr, name='optimizer2')
        trainable_var = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_var)
        grads1 = grads[:trainable_layer_len]
        grads2 = grads[trainable_layer_len:]
        # grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        train_op1 = optimizer1.apply_gradients(zip(grads1, trainable_var[:trainable_layer_len]))
        train_op2 = optimizer2.apply_gradients(zip(grads2, trainable_var[trainable_layer_len:]))
        self.train_op = tf.group(train_op1, train_op2)

    def train(self):
        image_dir = '/home/ty/data/video_saliency/train_all'
        label_dir = '/home/ty/data/video_saliency/train_all_gt2_revised'
        prior_dir = '/home/ty/data/video_saliency/train_all_prior'
        list_file_path = '/home/ty/data/video_saliency/train_all_seq.txt'

        # image_dir = '/home/ty/data/video_saliency/train_all'
        # label_dir = '/home/ty/data/video_saliency/train_all_gt2_revised'
        list_file = open(list_file_path)
        image_names = [line.strip() for line in list_file]

        # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
        dataset = ImageAndPriorSeqData(image_dir, label_dir, prior_dir, None, None,
                                       None,
                                       image_names, None, '.jpg', '.png', 550, 512, 1,
                                       self.batch_size,
                                       horizontal_flip=False)

        summary_op = tf.summary.merge_all()
        # dataset = ImageData(image_dir, label_dir, None, None, '.jpg', '.png', 550, 512, 5, horizontal_flip=True)
        summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)

        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_loss = []
        log_loss.append('Only spatial sub-network \n')


        for itr in range(10001):
            x, y = dataset.next_batch()
            feed_dict = {self.X: x[:, :, :, :3], self.Y: y}
            self.sess.run(self.train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, saliency, up, summary_str = self.sess.run([self.loss, self.final_saliency, self.up_fc8, summary_op], feed_dict=feed_dict)

                print ('step: %d, train_loss:%g' % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 2000 == 0:
                self.save(str(itr), time_str, log_loss)
                del log_loss[:]

    def train_ST(self):
        image_dir = '/home/ty/data/davis/480p'
        label_dir = '/home/ty/data/davis/GT'
        prior_dir = '/home/ty/data/davis/davis_flow_prior'
        davis_file = open('/home/ty/data/davis/davis_file.txt')
        image_names = [line.strip() for line in davis_file]

        validate_dir = '/home/ty/data/FBMS/FBMS_Testset2'
        validate_label_dir = '/home/ty/data/FBMS/GT_no_first'
        validate_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
        FBMS_file = open('/home/ty/data/FBMS/FBMS_file.txt')
        validate_names = [line.strip() for line in FBMS_file]
        # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
        dataset = ImageAndPriorData(image_dir, label_dir, prior_dir, validate_dir, validate_label_dir, validate_prior_dir, image_names,
                                    validate_names, '.jpg', '.png', 550, 512, 4, horizontal_flip=False)

        validate_x, validate_y = dataset.get_validate_images()
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)

        save_path = 'tempImages'
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        mae = 10.0
        for itr in range(30000):
            x, y = dataset.next_batch()
            feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y}
            self.sess.run(self.train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, saliency, up, summary_str = self.sess.run([self.loss, self.final_saliency, self.up_fc8, summary_op],
                                                                      feed_dict=feed_dict)
                print ('step: %d, train_loss:%g' % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 100 == 0:
                mae_tmp = self.evaluate(validate_x, validate_y)
                if mae > mae_tmp:
                    mae = mae_tmp
                    self.save('best', time_str)
                print ('during training MAE: ', mae_tmp)

            if itr % 2000 == 0:

                self.save(str(itr), time_str)

                if not os.path.exists(os.path.join(save_path, time_str)):
                    os.makedirs(os.path.join(save_path, time_str))

                for index in range(validate_x.shape[0]):
                    if index % 30 == 0:
                        test = validate_x[index, :, :, :3]
                        test_prior = validate_x[index, :, :, :]
                        test = test[np.newaxis, ...]
                        test_prior = test_prior[np.newaxis, ...]
                        feed_dict = {self.X: test, self.X_prior: test_prior}
                        saliency = self.sess.run(self.final_saliency, feed_dict=feed_dict)

                        saliency = saliency * 255
                        save_sal = saliency.astype(np.uint8)
                        save_img = Image.fromarray(save_sal[0, :, :, 0])
                        save_img.save(os.path.join(save_path, time_str, str(itr) + '_' + str(index) + '.png'))


    def train_ST_rnn(self):
        image_dir = '/home/ty/data/video_saliency/train_all'
        label_dir = '/home/ty/data/video_saliency/train_all_gt2'
        prior_dir = '/home/ty/data/video_saliency/train_all_prior'
        davis_file = open('/home/ty/data/video_saliency/train_all_seq.txt')
        image_names = [line.strip() for line in davis_file]

        validate_dir = '/home/ty/data/FBMS/FBMS_Testset2'
        validate_label_dir = '/home/ty/data/FBMS/GT_no_first'
        validate_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
        FBMS_file = open('/home/ty/data/FBMS/FBMS_file.txt')
        validate_names = [line.strip() for line in FBMS_file]
        # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
        dataset = ImageAndPriorSeqData(image_dir, label_dir, prior_dir, validate_dir, validate_label_dir,
                                       validate_prior_dir,
                                       image_names, validate_names, '.jpg', '.png', 550, 512, 1, 4,
                                       horizontal_flip=False)



        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)

        save_path = 'tempImages'
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        mae = 10.0
        for itr in range(16001):
            x, y = dataset.next_batch()
            feed_dict = {self.X: x[:, :, :, :3], self.Y: y}
            train_loss, saliency, pool4_ms_saliency, rnn_output, summary_str = self.sess.run(
                [self.loss, self.final_saliency, self.pool4_ms_saliency, self.rnn_output, summary_op],
                feed_dict=feed_dict)

            self.sess.run(self.train_op, feed_dict=feed_dict)

            if itr % 5 == 0:
                train_loss, saliency, rnn_output, summary_str = self.sess.run([self.loss, self.final_saliency, self.rnn_output, summary_op],
                                                                      feed_dict=feed_dict)
                print ('step: %d, train_loss:%g' % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 2000 == 0:

                self.save(str(itr), time_str)



    def test(self, ckpt_dir):
        test_dir = '/home/ty/data/FBMS/FBMS_Testset2'
        # test_label_dir = '/home/ty/data/FBMS/GT_no_first'
        test_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
        FBMS_file = open('/home/ty/data/FBMS/FBMS_file.txt')
        test_names = [line.strip() for line in FBMS_file]
        save_path = 'total_result/result_rnn'
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        save_path = save_path + '_' + time_str

        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.saver.restore(self.sess, ckpt_dir)

        for name in test_names:
            image = Image.open(os.path.join(test_dir, name + '.jpg'))
            prior = Image.open(os.path.join(test_prior_dir, name + '.png'))
            image, prior = resize_image_prior(image, prior)
            w, h = image.size
            input_prior, input = preprocess(image, prior)
            feed_dict = {self.X: input}
            saliency = self.sess.run(self.final_saliency, feed_dict=feed_dict)

            saliency = saliency * 255
            save_sal = saliency.astype(np.uint8)
            save_img = Image.fromarray(save_sal[0, :h, :w, 0])

            image_path = os.path.join(save_path, name + '.png')
            print('process:', image_path)

            if not os.path.exists(os.path.dirname(image_path)):
                os.makedirs(os.path.dirname(image_path))

            save_img.save(os.path.join(save_path, name + '.png'))

    def test_seq(self, ckpt_dir):
        test_dir = '/home/ty/data/FBMS/FBMS_Testset'
        # test_label_dir = '/home/ty/data/FBMS/GT_no_first'
        test_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
        FBMS_file = open('/home/ty/data/FBMS/FBMS_seq_file.txt')
        test_names = [line.strip() for line in FBMS_file]

        # test_dir = '/home/ty/data/SegTrack/SegTrackv2'
        # # test_label_dir = '/home/ty/data/FBMS/GT_no_first'
        # test_prior_dir = '/home/ty/data/SegTrack/SegTrackv2_flow_prior'
        # FBMS_file = open('/home/ty/data/SegTrack/SegTrack_seq_file.txt')
        # test_names = [line.strip() for line in FBMS_file]

        save_path = 'total_result/result_rnn'
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        save_path = save_path + '_' + time_str
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.saver.restore(self.sess, ckpt_dir)

        for name in test_names:
            images_path = name.split(',')
            batch_x = np.zeros([4, 512, 512, 4])
            batch_x_no_prior = np.zeros([4, 512, 512, 3])
            for i, image_name in enumerate(images_path):
                image = Image.open(os.path.join(test_dir, image_name + '.jpg'))
                prior = Image.open(os.path.join(test_prior_dir, image_name + '.png'))
                image, prior = resize_image_prior(image, prior)
                w, h = image.size
                input_prior, input = preprocess(image, prior)

                batch_x_no_prior[i] = input

            feed_dict = {self.X: batch_x_no_prior}
            saliency = self.sess.run(self.final_saliency, feed_dict=feed_dict)

            saliency = saliency * 255
            save_sal = saliency.astype(np.uint8)
            save_img = Image.fromarray(save_sal[3, :h, :w, 0])

            image_path = os.path.join(save_path, images_path[-1] + '.png')
            if not os.path.exists(os.path.dirname(image_path)):
                os.makedirs(os.path.dirname(image_path))

            save_img.save(image_path)

    def evaluate(self, x, y):
        mae = 0.0
        for index in range(x.shape[0]):
            test = x[index, :, :, :3]
            test_prior = x[index, :, :, :]
            test = test[np.newaxis, ...]
            test_prior = test_prior[np.newaxis, ...]
            feed_dict = {self.X: test, self.X_prior: test_prior}
            saliency = self.sess.run(self.final_saliency, feed_dict=feed_dict)
            mae += mean_averge_error(saliency, y[index, :, :, :])

        return mae / x.shape[0]

    def save(self, itr, network_name, log_list=[]):
        model_dir = os.path.join('models', network_name, itr)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        print ('save model:', os.path.join(model_dir, 'snap_model.ckpt'))
        self.saver.save(self.sess, os.path.join(model_dir, 'snap_model.ckpt'))

        if log_list:
            log_file = open(os.path.join('models', network_name, 'log_loss.txt'), 'a')
            for log in log_list:
                log_file.writelines(log + '\n')

            log_file.flush()
            log_file.close()

    def sampler(self, image):
        self.X_test = tf.placeholder(tf.float32, image.shape, name='test_image')

        conv1_1 = tf.nn.relu(self.conv2d(self.X_test, [3, 3, 3, 64], 'conv1_1'))
        conv1_2 = tf.nn.relu(self.conv2d(conv1_1, [3, 3, 64, 64], 'conv1_2'))
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        conv2_1 = tf.nn.relu(self.conv2d(pool1, [3, 3, 64, 128], 'conv2_1'))
        conv2_2 = tf.nn.relu(self.conv2d(conv2_1, [3, 3, 128, 128], 'conv2_2'))
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        conv3_1 = tf.nn.relu(self.conv2d(pool2, [3, 3, 128, 256], 'conv3_1'))
        conv3_2 = tf.nn.relu(self.conv2d(conv3_1, [3, 3, 256, 256], 'conv3_2'))
        conv3_3 = tf.nn.relu(self.conv2d(conv3_2, [3, 3, 256, 256], 'conv3_3'))
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        conv4_1 = tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 512], 'conv4_1'))
        conv4_2 = tf.nn.relu(self.conv2d(conv4_1, [3, 3, 512, 512], 'conv4_2'))
        conv4_3 = tf.nn.relu(self.conv2d(conv4_2, [3, 3, 512, 512], 'conv4_3'))
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')

        conv5_1 = tf.nn.relu(self.astro_conv2d(pool4, [3, 3, 512, 512], hole=2, name='conv5_1'))
        conv5_2 = tf.nn.relu(self.astro_conv2d(conv5_1, [3, 3, 512, 512], hole=2, name='conv5_2'))
        conv5_3 = tf.nn.relu(self.astro_conv2d(conv5_2, [3, 3, 512, 512], hole=2, name='conv5_3'))
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        fc6 = tf.nn.relu(self.astro_conv2d(pool5, [4, 4, 512, 4096], hole=4, name='fc6'))
        fc6_dropout = tf.nn.dropout(fc6, 0.5)

        fc7 = tf.nn.relu(self.astro_conv2d(fc6_dropout, [1, 1, 4096, 4096], hole=4, name='fc7'))
        fc7_dropout = tf.nn.dropout(fc7, 0.5)

        fc8 = self.conv2d(fc7_dropout, [1, 1, 4096, 1], 'fc8')
        up_fc8 = tf.image.resize_bilinear(fc8, [512, 512])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')
        up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [512, 512])
        self.final_saliency = tf.sigmoid(tf.add(up_pool4, up_fc8))

        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver()


    def load(self):
        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)


        # conv1 = tf.Variable('conv1_1_w:0')
        # conv2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64])
        # op = conv1.assign(tf.zeros(shape=[3, 3, 3, 64]))
        # print 'op:', self.sess.run(op)
        # print 'op:', self.sess.run(tf.assign(conv2, tf.zeros(shape=[3, 3, 64, 64], dtype=tf.float32)))

        ######## modify middle layer parameter #############
        # graph = tf.get_default_graph()
        # conv1 = graph.get_tensor_by_name('conv1_1_w:0')
        # op = tf.assign(conv1, tf.zeros(shape=[3, 3, 3, 64], dtype=tf.float32))
        # print 'op:', self.sess.run(op)
        # # print self.sess.run(conv1.initializer)
        # print 'print weights:', conv1.eval()

    def forward(self, image):
        # tf.initialize_all_variables().run()
        # self.sampler(image)
        return self.sess.run(self.final_saliency, feed_dict = {self.X_test: image})



if __name__ == '__main__':
    with tf.Session() as sess:
        # image_dir = '/home/ty/data/MSRA5000/images'
        # label_dir = '/home/ty/data/MSRA5000/GT2'
        # dataset = ImageData(image_dir, label_dir, None, None, '.jpg', '.png', 550, 512, 1, horizontal_flip=False)
        # x, y = dataset.next_batch()
        # parameter_path = 'models/2017-12-15 21:24:21/28000/snap_model.ckpt'
        # parameter_path = 'models/2017-12-16 22:49:43/28000/snap_model.ckpt'
        # parameter_path = 'models/2018-07-27 11:27:47/18000/snap_model.ckpt' #first stage training in spatial sub-network
        parameter_path = 'models/2018-07-31 09:28:38/10000/snap_model.ckpt' #second stage training in spatial sub-network
        # parameter_path = 'DCL_parameter/DCL_tensorflow.ckpt'
        # parameter_path = 'DCL_parameter/DCL_tensorflow_all_pool.ckpt'
        dcl = DCL(sess, 4, ckpt_dir=parameter_path)
        # dcl.train()
        # dcl.train_ST_rnn()
        dcl.test(parameter_path)
        # dcl.test('models/2017-12-10 21:18:26/28000/snap_model.ckpt')
        # dcl.sampler(x)
        # dcl.load()
        # r = dcl.forward(x)
        # plt.imshow(r[0, :, :, 0])
        # plt.show()
        #
        # print np.shape(in_)

        # dcl.load()
        # dcl.train()
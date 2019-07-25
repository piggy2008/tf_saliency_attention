import tensorflow as tf
import numpy as np
import math
import random
from PIL import Image
from cell import ConvLSTMCell
import cv2
from matplotlib import pyplot as plt
from image_data_loader import ImageData, ImageAndPriorData, ImageAndPriorSeqData, ImageAndFlowSeqData
import os
import time
from utils import preprocess, preprocess2, preprocess3
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from utils import MaxMinNormalization

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

def resize_image_prior2(image, prior, max_shape=512):
    w, h = image.size

    image = image.resize([512, 512])
    prior = prior.resize([512, 512])

    return image, prior

class VideoSailency(object):

    def __init__(self, sess, batch_size, drop_path=False, drop_path_type='default', image_size=550, crop_size=512, prior_type='prior', lr=0.00001, ckpt_dir='./parameters'):
        self.sess = sess
        self.ckpt_dir = ckpt_dir
        self.lr = lr
        self.batch_size = batch_size
        self.prior_type = prior_type
        self.drop_path = drop_path
        self.crop_size = crop_size
        self.image_size = image_size
        if drop_path:
            self.build_ST_RNN_drop_path()
        else:
            self.build_ST_C3D()

        # drop_path_type: choose1, choose2 and default
        # choose 1: one path
        # choose 2: two paths
        # default: random choose 1, 2 or 3 paths to train
        self.drop_path_type = drop_path_type

        # self.build_ST()
        # self.build_ST_RNN()

    def conv2d(self, x, w_shape, name):
        w = tf.Variable(tf.truncated_normal(shape=w_shape), dtype=tf.float32, name=name + '_w')
        b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, w_shape[-1]]), dtype=tf.float32, name=name + '_b')
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b

    def conv3d(self, x, w_shape, name):
        w = tf.Variable(tf.truncated_normal(shape=w_shape), dtype=tf.float32, name=name + '_w')
        b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, 1, w_shape[-1]]), dtype=tf.float32, name=name + '_b')
        return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME') + b

    def astro_conv2d(self, x, w_shape, hole=2, name='astro_conv'):
        w = tf.Variable(tf.truncated_normal(shape=w_shape), dtype=tf.float32, name=name + '_w')
        b = tf.Variable(tf.truncated_normal(shape=[1, 1, 1, w_shape[-1]]), dtype=tf.float32, name=name + '_b')
        return tf.nn.atrous_conv2d(x, w, rate=hole, padding='SAME') + b

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    def build_ST_C3D(self):
        ############### Input ###############
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 3], name='rgb_image')

        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 1], name='gt')
        if (self.prior_type == 'prior'):
            self.X_prior = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 4],
                                          name='rgb_prior_image')
        else:
            self.X_prior = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 3],
                                          name='rgb_flow_image')

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
        # rnn_output_fc8 = self.rnn_cell(fc8, 'fc8')
        # fc8 = tf.add(rnn_output_fc8, fc8)

        up_fc8 = tf.image.resize_bilinear(fc8, [128, 128])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')
        up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [128, 128])

        # pool3_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 128], 'pool3_conv')), 0.5)
        # pool3_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_conv, [1, 1, 128, 128], 'pool3_fc')), 0.5)
        # pool3_ms_saliency = self.conv2d(pool3_fc, [1, 1, 128, 1], 'pool3_ms_saliency')
        # up_pool3 = tf.image.resize_bilinear(pool3_ms_saliency, [self.crop_size, self.crop_size])
        # rnn_output_pool4 = self.rnn_cell(pool4_ms_saliency, 'pool4')
        # pool4_ms_saliency = tf.add(rnn_output_pool4, pool4_ms_saliency)

        # final_saliency_r1 = tf.add(up_pool4, up_fc8)

        ############### R2 ###############
        if (self.prior_type == 'prior'):
            conv1_1_r2 = tf.nn.relu(self.conv2d(self.X_prior, [3, 3, 4, 64], 'conv1_1_r2'))
        else:
            conv1_1_r2 = tf.nn.relu(self.conv2d(self.X_prior, [3, 3, 3, 64], 'conv1_1_r2'))

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
        up_fc8_r2 = tf.image.resize_bilinear(fc8_r2, [128, 128])

        pool4_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_r2, [3, 3, 512, 128], 'pool4_conv_r2')), 0.5)
        pool4_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv_r2, [1, 1, 128, 128], 'pool4_fc_r2')), 0.5)
        pool4_ms_saliency_r2 = self.conv2d(pool4_fc_r2, [1, 1, 128, 1], 'pool4_ms_saliency_r2')
        up_pool4_r2 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [128, 128])

        # pool3_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_r2, [3, 3, 256, 128], 'pool3_conv_r2')), 0.5)
        # pool3_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_conv_r2, [1, 1, 128, 128], 'pool3_fc_r2')), 0.5)
        # pool3_ms_saliency_r2 = self.conv2d(pool3_fc_r2, [1, 1, 128, 1], 'pool3_ms_saliency_r2')
        # up_pool3_r2 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [self.crop_size, self.crop_size])

        # final_saliency_r2 = tf.add(up_pool4_r2, up_fc8_r2)

        ########## rnn fusion ############

        inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2, up_fc8, up_fc8_r2], axis=3), 0)
        # cell = ConvLSTMCell([128, 128], 1, [3, 3])
        # outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, scope='rnn')
        # rnn_output = tf.squeeze(outputs, axis=0)
        # up_rnn_output = tf.image.resize_bilinear(rnn_output, [self.crop_size, self.crop_size])
        # up2_rnn_output = tf.image.resize_bilinear(rnn_output, [128, 128])
        ########## C3D fusion ############

        # inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2, up_fc8, up_fc8_r2], axis=3), 0)
        C3D_outputs = self.conv3d(inputs, [3, 3, 3, 4, 1], name='3D_conv')
        C3D_output = tf.squeeze(C3D_outputs, axis=0)
        # up_C3D_output = tf.image.resize_bilinear(C3D_output, [self.crop_size, self.crop_size])
        # up2_C3D_output = tf.image.resize_bilinear(C3D_output, [128, 128])
        ########### ST fusion ############
        pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3, name='concat_pool4')
        pool4_saliency_ST = self.conv2d(pool4_saliency_cancat, [1, 1, 2, 1], 'pool4_saliency_ST')
        # up_pool4_ST = tf.image.resize_bilinear(pool4_saliency_ST, [self.crop_size, self.crop_size])

        # pool3_saliency_cancat = tf.concat([pool3_ms_saliency, pool3_ms_saliency_r2], 3, name='concat_pool3')
        # pool3_saliency_ST = self.conv2d(pool3_saliency_cancat, [1, 1, 2, 1], 'pool3_saliency_ST')
        # up_pool3_ST = tf.image.resize_bilinear(pool3_saliency_ST, [self.crop_size, self.crop_size])

        fc8_concat = tf.concat([fc8, fc8_r2], 3, name='concat_fc8')
        fc8_saliency_ST = self.conv2d(fc8_concat, [1, 1, 2, 1], 'fc8_saliency_ST')
        # up_fc8_ST = tf.image.resize_bilinear(fc8_saliency_ST, [self.crop_size, self.crop_size])

        # pool4_fc8_concat = tf.concat([up_pool3_ST, up_pool4_ST, up_fc8_ST], axis=3)
        # pool4_fc8_combine = self.conv2d(pool4_fc8_concat, [1, 1, 3, 1], 'pool4_fc8')
        # pool4_fc8_combine = tf.add(pool3_saliency_ST, pool4_saliency_ST)
        pool4_fc8_combine = tf.add(pool4_saliency_ST, fc8_saliency_ST)
        up2_pool4_fc8_combine = tf.image.resize_bilinear(pool4_fc8_combine, [128, 128])

        # up_pool4_fc8_combine = tf.add(up_pool3_ST, up_pool4_ST)
        # up_pool4_fc8_combine = tf.add(up_pool4_fc8_combine, up_fc8_ST)

        # final_saliency = tf.add(up_pool4_ST, up_fc8_ST)
        # final_saliency = tf.add(final_saliency, up_pool4_r2)
        # final_saliency = tf.add(final_saliency, up_fc8_r2)
        # final_saliency = tf.add(pool4_fc8_combine, rnn_output)
        # final_saliency = tf.add(final_saliency, C3D_output)

        ########### attetion fusion ############
        # motion_cancat = tf.concat([up2_pool4_fc8_combine, up2_C3D_output, up2_rnn_output], axis=3)
        # # attention_first = tf.nn.dropout(tf.nn.relu(self.conv2d(motion_cancat, [3, 3, 3, 128], 'attention_conv1')), 0.5)
        # attention_first = tf.nn.relu(self.conv2d(motion_cancat, [3, 3, 3, 256], 'attention_conv1'))
        # attention_second = tf.nn.softmax(self.conv2d(attention_first, [1, 1, 256, 3], 'attention_conv2'))
        #
        # up_motion_cancat = tf.concat([up_pool4_fc8_combine, up_C3D_output, up_rnn_output], axis=3)
        # up_attention = tf.image.resize_bilinear(attention_second, [self.crop_size, self.crop_size])
        #
        # final_fusion = tf.multiply(up_motion_cancat, up_attention)
        # final_saliency = tf.reduce_sum(final_fusion, axis=3, keep_dims=True)
        # ave_num = tf.constant(3.0, dtype=tf.float32, shape=[self.batch_size, self.crop_size, self.crop_size, 1])
        # final_saliency = tf.div(final_saliency, ave_num)
        final_saliency = tf.add(up2_pool4_fc8_combine, C3D_output)
        final_saliency = tf.image.resize_bilinear(final_saliency, [512, 512])
        # final_saliency = tf.reduce_sum(final_fusion, axis=3, keep_dims=True)
        self.final_saliency = tf.sigmoid(final_saliency)
        self.up_fc8 = up_fc8
        self.rnn_output = final_saliency
        self.saver = tf.train.Saver()

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=self.Y),
                                   name='loss')
        # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=final_saliency, targets=self.Y, pos_weight=0.12), name='loss')

        # self.loss_rnn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C3D_output, labels=self.Y),
        #                            name='loss2')
        # tf.summary.scalar('entropy', self.loss + 0.1 * self.loss_rnn)
        tf.summary.scalar('entropy', self.loss)
        trainable_var = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
        # grads = optimizer.compute_gradients(self.loss + 1 * self.loss_rnn, var_list=trainable_var)
        grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        # grads = optimizer.compute_gradients(self.loss + 0.5 * self.loss_rnn, var_list=trainable_var[-22:])

        # optimizer2 = tf.train.MomentumOptimizer(self.lr, 0.9)
        # grads = optimizer2.compute_gradients(self.loss + 0.5 * self.loss_rnn, var_list=trainable_var[-22:])
        # grads = optimizer2.compute_gradients(self.loss, var_list=trainable_var)
        self.train_op = optimizer.apply_gradients(grads)

    def build_ST_RNN(self):
        ############### Input ###############
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 3], name='rgb_image')

        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 1], name='gt')
        if (self.prior_type == 'prior'):
            self.X_prior = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 4], name='rgb_prior_image')
        else:
            self.X_prior = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 3], name='rgb_flow_image')

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
        # rnn_output_fc8 = self.rnn_cell(fc8, 'fc8')
        # fc8 = tf.add(rnn_output_fc8, fc8)

        up_fc8 = tf.image.resize_bilinear(fc8, [128, 128])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')
        up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [128, 128])

        pool3_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 128], 'pool3_conv')), 0.5)
        pool3_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_conv, [1, 1, 128, 128], 'pool3_fc')), 0.5)
        pool3_ms_saliency = self.conv2d(pool3_fc, [1, 1, 128, 1], 'pool3_ms_saliency')
        up_pool3 = tf.image.resize_bilinear(pool3_ms_saliency, [self.crop_size, self.crop_size])
        # rnn_output_pool4 = self.rnn_cell(pool4_ms_saliency, 'pool4')
        # pool4_ms_saliency = tf.add(rnn_output_pool4, pool4_ms_saliency)


        # final_saliency_r1 = tf.add(up_pool4, up_fc8)

        ############### R2 ###############
        if (self.prior_type == 'prior'):
            conv1_1_r2 = tf.nn.relu(self.conv2d(self.X_prior, [3, 3, 4, 64], 'conv1_1_r2'))
        else:
            conv1_1_r2 = tf.nn.relu(self.conv2d(self.X_prior, [3, 3, 3, 64], 'conv1_1_r2'))


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
        up_fc8_r2 = tf.image.resize_bilinear(fc8_r2, [128, 128])

        pool4_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_r2, [3, 3, 512, 128], 'pool4_conv_r2')), 0.5)
        pool4_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv_r2, [1, 1, 128, 128], 'pool4_fc_r2')), 0.5)
        pool4_ms_saliency_r2 = self.conv2d(pool4_fc_r2, [1, 1, 128, 1], 'pool4_ms_saliency_r2')
        up_pool4_r2 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [128, 128])

        pool3_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_r2, [3, 3, 256, 128], 'pool3_conv_r2')), 0.5)
        pool3_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_conv_r2, [1, 1, 128, 128], 'pool3_fc_r2')), 0.5)
        pool3_ms_saliency_r2 = self.conv2d(pool3_fc_r2, [1, 1, 128, 1], 'pool3_ms_saliency_r2')
        # up_pool3_r2 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [self.crop_size, self.crop_size])

        # final_saliency_r2 = tf.add(up_pool4_r2, up_fc8_r2)

        ########## rnn fusion ############

        inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2, up_fc8, up_fc8_r2], axis=3), 0)
        cell = ConvLSTMCell([128, 128], 1, [3, 3])
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, scope='rnn')
        rnn_output = tf.squeeze(outputs, axis=0)
        up_rnn_output = tf.image.resize_bilinear(rnn_output, [self.crop_size, self.crop_size])
        up2_rnn_output = tf.image.resize_bilinear(rnn_output, [128, 128])
        ########## C3D fusion ############

        # inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2, up_fc8, up_fc8_r2], axis=3), 0)
        C3D_outputs = self.conv3d(inputs, [3, 3, 3, 4, 1], name='3D_conv')
        C3D_output = tf.squeeze(C3D_outputs, axis=0)
        up_C3D_output = tf.image.resize_bilinear(C3D_output, [self.crop_size, self.crop_size])
        up2_C3D_output = tf.image.resize_bilinear(C3D_output, [128, 128])
        ########### ST fusion ############
        pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3, name='concat_pool4')
        pool4_saliency_ST = self.conv2d(pool4_saliency_cancat, [1, 1, 2, 1], 'pool4_saliency_ST')
        up_pool4_ST = tf.image.resize_bilinear(pool4_saliency_ST, [self.crop_size, self.crop_size])

        pool3_saliency_cancat = tf.concat([pool3_ms_saliency, pool3_ms_saliency_r2], 3, name='concat_pool3')
        pool3_saliency_ST = self.conv2d(pool3_saliency_cancat, [1, 1, 2, 1], 'pool3_saliency_ST')
        up_pool3_ST = tf.image.resize_bilinear(pool3_saliency_ST, [self.crop_size, self.crop_size])

        fc8_concat = tf.concat([fc8, fc8_r2], 3, name='concat_fc8')
        fc8_saliency_ST = self.conv2d(fc8_concat, [1, 1, 2, 1], 'fc8_saliency_ST')
        up_fc8_ST = tf.image.resize_bilinear(fc8_saliency_ST, [self.crop_size, self.crop_size])

        # pool4_fc8_concat = tf.concat([up_pool3_ST, up_pool4_ST, up_fc8_ST], axis=3)
        # pool4_fc8_combine = self.conv2d(pool4_fc8_concat, [1, 1, 3, 1], 'pool4_fc8')
        pool4_fc8_combine = tf.add(pool3_saliency_ST, pool4_saliency_ST)
        pool4_fc8_combine = tf.add(pool4_fc8_combine, fc8_saliency_ST)
        up2_pool4_fc8_combine = tf.image.resize_bilinear(pool4_fc8_combine, [128, 128])

        up_pool4_fc8_combine = tf.add(up_pool3_ST, up_pool4_ST)
        up_pool4_fc8_combine = tf.add(up_pool4_fc8_combine, up_fc8_ST)

        # final_saliency = tf.add(up_pool4_ST, up_fc8_ST)
        # final_saliency = tf.add(final_saliency, up_pool4_r2)
        # final_saliency = tf.add(final_saliency, up_fc8_r2)
        # final_saliency = tf.add(pool4_fc8_combine, rnn_output)
        # final_saliency = tf.add(final_saliency, C3D_output)

        ########### attetion fusion ############
        motion_cancat = tf.concat([up2_pool4_fc8_combine, up2_C3D_output, up2_rnn_output], axis=3)
        # attention_first = tf.nn.dropout(tf.nn.relu(self.conv2d(motion_cancat, [3, 3, 3, 128], 'attention_conv1')), 0.5)
        attention_first = tf.nn.relu(self.conv2d(motion_cancat, [3, 3, 3, 256], 'attention_conv1'))
        attention_second = tf.nn.softmax(self.conv2d(attention_first, [1, 1, 256, 3], 'attention_conv2'))

        up_motion_cancat = tf.concat([up_pool4_fc8_combine, up_C3D_output, up_rnn_output], axis=3)
        up_attention = tf.image.resize_bilinear(attention_second, [self.crop_size, self.crop_size])

        final_fusion = tf.multiply(up_motion_cancat, up_attention)
        final_saliency = tf.reduce_sum(final_fusion, axis=3, keep_dims=True)
        ave_num = tf.constant(3.0, dtype=tf.float32, shape=[self.batch_size, self.crop_size, self.crop_size, 1])
        final_saliency = tf.div(final_saliency, ave_num)

        self.final_saliency = tf.sigmoid(final_saliency)
        self.up_fc8 = up_fc8
        self.rnn_output = final_saliency
        self.saver = tf.train.Saver()

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=self.Y), name='loss')
        # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=final_saliency, targets=self.Y, pos_weight=0.12), name='loss')

        # self.loss_rnn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C3D_output, labels=self.Y),
        #                            name='loss2')
        # tf.summary.scalar('entropy', self.loss + 0.1 * self.loss_rnn)
        tf.summary.scalar('entropy', self.loss)
        trainable_var = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
        # grads = optimizer.compute_gradients(self.loss + 1 * self.loss_rnn, var_list=trainable_var)
        grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        # grads = optimizer.compute_gradients(self.loss + 0.5 * self.loss_rnn, var_list=trainable_var[-22:])

        # optimizer2 = tf.train.MomentumOptimizer(self.lr, 0.9)
        # grads = optimizer2.compute_gradients(self.loss + 0.5 * self.loss_rnn, var_list=trainable_var[-22:])
        # grads = optimizer2.compute_gradients(self.loss, var_list=trainable_var)
        self.train_op = optimizer.apply_gradients(grads)

    def build_ST_RNN_drop_path(self):
        ############### Input ###############
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 3], name='rgb_image')

        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 1], name='gt')
        if (self.prior_type == 'prior'):
            self.X_prior = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 4], name='rgb_prior_image')
        else:
            self.X_prior = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 3], name='rgb_flow_image')


        self.fusion_weight = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 3], name='fusion_weight')
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
        # rnn_output_fc8 = self.rnn_cell(fc8, 'fc8')
        # fc8 = tf.add(rnn_output_fc8, fc8)

        up_fc8 = tf.image.resize_bilinear(fc8, [self.crop_size, self.crop_size])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')
        up_pool4 = tf.image.resize_bilinear(pool4_ms_saliency, [self.crop_size, self.crop_size])

        pool3_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3, [3, 3, 256, 128], 'pool3_conv')), 0.5)
        pool3_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_conv, [1, 1, 128, 128], 'pool3_fc')), 0.5)
        pool3_ms_saliency = self.conv2d(pool3_fc, [1, 1, 128, 1], 'pool3_ms_saliency')
        # up_pool3 = tf.image.resize_bilinear(pool3_ms_saliency, [self.crop_size, self.crop_size])
        # rnn_output_pool4 = self.rnn_cell(pool4_ms_saliency, 'pool4')
        # pool4_ms_saliency = tf.add(rnn_output_pool4, pool4_ms_saliency)



        # final_saliency_r1 = tf.add(up_pool4, up_fc8)

        ############### R2 ###############
        if (self.prior_type == 'prior'):
            conv1_1_r2 = tf.nn.relu(self.conv2d(self.X_prior, [3, 3, 4, 64], 'conv1_1_r2'))
        else:
            conv1_1_r2 = tf.nn.relu(self.conv2d(self.X_prior, [3, 3, 3, 64], 'conv1_1_r2'))


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
        up_fc8_r2 = tf.image.resize_bilinear(fc8_r2, [self.crop_size, self.crop_size])

        pool4_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_r2, [3, 3, 512, 128], 'pool4_conv_r2')), 0.5)
        pool4_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv_r2, [1, 1, 128, 128], 'pool4_fc_r2')), 0.5)
        pool4_ms_saliency_r2 = self.conv2d(pool4_fc_r2, [1, 1, 128, 1], 'pool4_ms_saliency_r2')
        up_pool4_r2 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [self.crop_size, self.crop_size])
        # final_saliency_r2 = tf.add(up_pool4_r2, up_fc8_r2)

        pool3_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_r2, [3, 3, 256, 128], 'pool3_conv_r2')), 0.5)
        pool3_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_conv_r2, [1, 1, 128, 128], 'pool3_fc_r2')), 0.5)
        pool3_ms_saliency_r2 = self.conv2d(pool3_fc_r2, [1, 1, 128, 1], 'pool3_ms_saliency_r2')
        # up_pool3_r2 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [self.crop_size, self.crop_size])

        ########## rnn fusion ############

        inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2, up_fc8, up_fc8_r2], axis=3), 0)
        cell = ConvLSTMCell([self.crop_size, self.crop_size], 1, [3, 3])
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, scope='rnn')
        rnn_output = tf.squeeze(outputs, axis=0)

        ########## C3D fusion ############

        # inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2, up_fc8, up_fc8_r2], axis=3), 0)
        C3D_outputs = self.conv3d(inputs, [3, 3, 3, 4, 1], name='3D_conv')
        C3D_output = tf.squeeze(C3D_outputs, axis=0)

        ########### ST fusion ############
        pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3, name='concat_pool4')
        pool4_saliency_ST = self.conv2d(pool4_saliency_cancat, [1, 1, 2, 1], 'pool4_saliency_ST')
        up_pool4_ST = tf.image.resize_bilinear(pool4_saliency_ST, [self.crop_size, self.crop_size])

        pool3_saliency_cancat = tf.concat([pool3_ms_saliency, pool3_ms_saliency_r2], 3, name='concat_pool3')
        pool3_saliency_ST = self.conv2d(pool3_saliency_cancat, [1, 1, 2, 1], 'pool3_saliency_ST')
        up_pool3_ST = tf.image.resize_bilinear(pool3_saliency_ST, [self.crop_size, self.crop_size])

        fc8_concat = tf.concat([fc8, fc8_r2], 3, name='concat_fc8')
        fc8_saliency_ST = self.conv2d(fc8_concat, [1, 1, 2, 1], 'fc8_saliency_ST')
        up_fc8_ST = tf.image.resize_bilinear(fc8_saliency_ST, [self.crop_size, self.crop_size])


        # pool4_fc8_concat = tf.concat([up_pool3_ST, up_pool4_ST, up_fc8_ST], axis=3)
        # pool4_fc8_combine = self.conv2d(pool4_fc8_concat, [1, 1, 3, 1], 'pool4_fc8')
        pool4_fc8_combine = tf.add(up_pool3_ST, up_pool4_ST)
        pool4_fc8_combine = tf.add(pool4_fc8_combine, up_fc8_ST)
        # final_saliency = tf.add(up_pool4_ST, up_fc8_ST)
        # final_saliency = tf.add(final_saliency, up_pool4_r2)
        # final_saliency = tf.add(final_saliency, up_fc8_r2)

        # drop path process
        fusion = tf.concat([pool4_fc8_combine, rnn_output, C3D_output], axis=3)
        fusion_drop_path = tf.multiply(fusion, self.fusion_weight)
        final_saliency = tf.reduce_sum(fusion_drop_path, axis=3, keep_dims=True)
        ave_num = tf.constant(3.0, dtype=tf.float32, shape=[self.batch_size, self.crop_size, self.crop_size, 1])
        final_saliency = tf.div(final_saliency, ave_num)

        self.final_saliency = tf.sigmoid(final_saliency)
        self.up_fc8 = up_fc8
        self.rnn_output = final_saliency
        self.saver = tf.train.Saver()

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=self.Y), name='loss')
        # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=final_saliency, targets=self.Y, pos_weight=0.12), name='loss')
        # self.loss_rnn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C3D_output, labels=self.Y),
        #                            name='loss2')
        # tf.summary.scalar('entropy', self.loss + 0.5 * self.loss_rnn)
        tf.summary.scalar('entropy', self.loss)
        optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
        trainable_var = tf.trainable_variables()
        # grads = optimizer.compute_gradients(self.loss + 0.5 * self.loss_rnn, var_list=trainable_var)
        # grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)

        # optimizer2 = tf.train.MomentumOptimizer(self.lr, 0.99)
        grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        # grads = optimizer2.compute_gradients(self.loss, var_list=trainable_var[-46:])
        self.train_op = optimizer.apply_gradients(grads)


    def train_ST_rnn(self, train_dir, label_dir, prior_dir, list_file_path, iter_num=4000):

        list_file = open(list_file_path)
        image_names = [line.strip() for line in list_file]


        # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
        dataset = ImageAndPriorSeqData(train_dir, label_dir, prior_dir, None, None,
                                       None,
                                       image_names, None, '.jpg', '.png', self.image_size, self.crop_size, 1, self.batch_size,
                                       horizontal_flip=False)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)
        random_path = -1
        # save_path = 'tempImages'
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_loss = []
        log_loss.append('prior type:' + str(self.prior_type) + '\n')
        log_loss.append('starting lr:' + str(self.lr) + '\n')
        if self.drop_path:
            log_loss.append('drop path is introduced\n')
            log_loss.append('drop path type is ' + self.drop_path_type + '\n')
        for itr in range(iter_num + 1):
            x, y = dataset.next_batch()
            # feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y}
            if self.drop_path:
                if self.drop_path_type == 'choose1':
                    # choose one path to backpro (totally 3 path: static, C3D, RNN)
                    # totally 3 paths, choose one to backpro
                    weight = np.zeros([x.shape[0], x.shape[1], x.shape[2], 3], dtype=np.float16)
                    random_path = random.randint(0, 2)
                    weight[:, :, :, random_path] = np.ones([x.shape[0], x.shape[1], x.shape[2]], dtype=np.float16)
                    feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y, self.fusion_weight: weight}
                elif self.drop_path_type == 'choose2':
                    # choose 2 paths to backpro (totally 3 path: static, C3D, RNN)
                    weight = np.ones([x.shape[0], x.shape[1], x.shape[2], 3], dtype=np.float16)
                    random_path = random.randint(0, 2)
                    weight[:, :, :, random_path] = np.zeros([x.shape[0], x.shape[1], x.shape[2]], dtype=np.float16)
                    feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y,
                                 self.fusion_weight: weight}
                else:
                    # this senario can randomly choose 1, 2 or 3 paths to backpro
                    path_num = 3
                    # path_num = 3
                    if path_num == 1:
                        weight = np.zeros([x.shape[0], x.shape[1], x.shape[2], 3], dtype=np.float16)
                        random_path = random.randint(0, 2)
                        weight[:, :, :, random_path] = np.ones([x.shape[0], x.shape[1], x.shape[2]], dtype=np.float16)
                        feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y,
                                     self.fusion_weight: weight}
                    elif path_num == 2:
                        weight = np.ones([x.shape[0], x.shape[1], x.shape[2], 3], dtype=np.float16)
                        random_path = random.randint(0, 2)
                        # weight[:, :, :, random_path] = np.zeros([x.shape[0], x.shape[1], x.shape[2]], dtype=np.float16)
                        # weight not set to 0 from beginning. firstly set to 0.8, then 0.5, last 0.2, final 0
                        step = 0.0
                        weight[:, :, :, random_path] = np.ones([x.shape[0], x.shape[1], x.shape[2]], dtype=np.float16) * step

                        feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y,
                                     self.fusion_weight: weight}
                    else:
                        weight = np.ones([x.shape[0], x.shape[1], x.shape[2], 3], dtype=np.float16)
                        feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y,
                                     self.fusion_weight: weight}

            else:
                feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y}

            self.sess.run(self.train_op, feed_dict=feed_dict)

            if itr % 5 == 0:
                train_loss, saliency, rnn_output, summary_str = self.sess.run(
                    [self.loss, self.final_saliency, self.rnn_output, summary_op],
                    feed_dict=feed_dict)
                print ('step: %d, train_loss:%g' % (itr, train_loss))
                if self.drop_path:
                    print ('step: %d, path_num:%d, random_path: %d' % (itr, path_num, random_path))
                log_loss.append('step: %d, train_loss:%g' % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 2000 == 0:
                self.save(str(itr), time_str, log_loss)
                del log_loss[:]

    def train_flow_ST_rnn(self, train_dir, label_dir, flow_dir, list_file_path):

        list_file = open(list_file_path)
        image_names = [line.strip() for line in list_file]


        # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
        dataset = ImageAndFlowSeqData(train_dir, label_dir, flow_dir, None, None,
                                       None,
                                       image_names, None, '.jpg', '.png', self.image_size, self.crop_size, 1, self.batch_size,
                                       horizontal_flip=False)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)

        # save_path = 'tempImages'
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_loss = []
        log_loss.append(self.prior_type + '\n')
        if self.drop_path:
            log_loss.append('drop path is introduced\n')
            log_loss.append('drop path type is ', self.drop_path_type, '\n')

        for itr in range(26001):
            x, y = dataset.next_batch()
            if self.drop_path:
                weight = np.zeros([x.shape[0], x.shape[1], x.shape[2], 3], dtype=np.float16)
                random_path = random.randint(0, 2)   # totally 3 paths, choose one to backpro
                weight[:, :, :, random_path] = np.ones([x.shape[0], x.shape[1], x.shape[2]], dtype=np.float16)
                feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x[:, :, :, 3:], self.Y: y, self.fusion_weight: weight}
            else:
                feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x[:, :, :, 3:], self.Y: y}
            # self.sess.run([self.loss, self.final_saliency, self.up_fc8, self.rnn_output, summary_op],
            #     feed_dict=feed_dict)

            self.sess.run(self.train_op, feed_dict=feed_dict)

            if itr % 5 == 0:
                train_loss, saliency, rnn_output, summary_str = self.sess.run(
                    [self.loss, self.final_saliency, self.rnn_output, summary_op],
                    feed_dict=feed_dict)
                print ('step: %d, train_loss:%g' % (itr, train_loss))
                log_loss.append('step: %d, train_loss:%g' % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 2000 == 0:
                self.save(str(itr), time_str, log_loss)
                del log_loss[:]


    def test_seq(self, test_dir, test_prior_dir, list_file_path, save_path):

        list_file = open(list_file_path)
        test_names = [line.strip() for line in list_file]


        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        save_path = save_path + '_' + time_str

        self.init = tf.global_variables_initializer()
        # self.sess.run(self.init)
        # self.saver.restore(self.sess, ckpt_dir)
        # load parameters
        self.load()

        for name in test_names:
            images_path = name.split(',')
            batch_x = np.zeros([4, 512, 512, 4])
            batch_x_no_prior = np.zeros([4, 512, 512, 3])
            for i, image_name in enumerate(images_path):
                image = Image.open(os.path.join(test_dir, image_name + '.jpg'))
                prior = Image.open(os.path.join(test_prior_dir, image_name + '.png'))
                w, h = image.size
                image, prior = resize_image_prior2(image, prior)

                input_prior, input = preprocess3(image, prior)

                batch_x_no_prior[i] = input
                batch_x[i] = input_prior

            if self.drop_path:
                weight = np.ones([4, 512, 512, 3], dtype=np.float16)
                feed_dict = {self.X: batch_x_no_prior, self.X_prior: batch_x, self.fusion_weight: weight}
            else:
                feed_dict = {self.X: batch_x_no_prior, self.X_prior: batch_x}

            # feed_dict = {self.X: batch_x_no_prior, self.X_prior: batch_x}
            start = time.clock()
            saliency = self.sess.run(self.final_saliency, feed_dict=feed_dict)
            end = time.clock()

            # plt.subplot(2, 1, 1)
            # plt.imshow(crop_final[3, :, :, 0])
            #
            # plt.subplot(2, 1, 2)
            # plt.imshow(saliency[3, :, :, 0])
            #
            # plt.show()

            saliency = saliency * 255
            save_sal = saliency.astype('float')
            save_sal = MaxMinNormalization(save_sal, save_sal.max(), save_sal.min()) * 255.0
            save_sal = save_sal.astype('uint8')
            save_img = Image.fromarray(save_sal[3, :h, :w, 0])

            image_path = os.path.join(save_path, images_path[-1] + '.png')
            print ('process:', image_path)
            print('time:', end - start)
            if not os.path.exists(os.path.dirname(image_path)):
                os.makedirs(os.path.dirname(image_path))

            save_img.save(image_path)

    def test_flow_seq(self, test_dir, test_prior_dir, list_file_path, save_path):

        list_file = open(list_file_path)
        test_names = [line.strip() for line in list_file]


        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        save_path = save_path + '_' + time_str

        self.init = tf.global_variables_initializer()
        # self.sess.run(self.init)
        # self.saver.restore(self.sess, ckpt_dir)
        # load parameters
        self.load()

        for name in test_names:
            images_path = name.split(',')
            batch_x = np.zeros([4, 512, 512, 3])
            batch_x_no_prior = np.zeros([4, 512, 512, 3])
            for i, image_name in enumerate(images_path):
                image = Image.open(os.path.join(test_dir, image_name + '.jpg'))
                flow = Image.open(os.path.join(test_prior_dir, image_name + '.jpg'))
                image, flow = resize_image_prior(image, flow)
                w, h = image.size
                input = preprocess2(image)
                input_flow = preprocess2(flow)

                batch_x_no_prior[i] = input
                batch_x[i] = input_flow
            if self.drop_path:
                weight = np.ones([4, 512, 512, 3], dtype=np.float16)
                feed_dict = {self.X: batch_x_no_prior, self.X_prior: batch_x, self.fusion_weight: weight}
            else:
                feed_dict = {self.X: batch_x_no_prior, self.X_prior: batch_x}
            saliency = self.sess.run(self.final_saliency, feed_dict=feed_dict)

            saliency = saliency * 255
            save_sal = saliency.astype(np.uint8)
            save_img = Image.fromarray(save_sal[3, :h, :w, 0])

            image_path = os.path.join(save_path, images_path[-1] + '.png')
            print ('process:', image_path)
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

    def load(self):
        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)




    def forward(self, image):
        # tf.initialize_all_variables().run()
        # self.sampler(image)
        return self.sess.run(self.final_saliency, feed_dict = {self.X_test: image})




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
from utils import preprocess, preprocess2
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

class VideoSailency(object):

    def __init__(self, sess, batch_size, drop_path=False, image_size=550, crop_size=512, lr=0.00001, ckpt_dir='./parameters'):
        self.sess = sess
        self.ckpt_dir = ckpt_dir
        self.lr = lr
        self.batch_size = batch_size
        self.drop_path = drop_path
        self.crop_size = crop_size
        self.image_size = image_size

        self.build_ST_RNN()

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


    def build_ST_RNN(self):
        ############### Input ###############
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 4], name='rgb_prior_image')

        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 1], name='gt')

        size = 512

        conv1_1_r2 = tf.nn.relu(self.conv2d(self.X, [3, 3, 4, 64], 'conv1_1_r2'))
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


        pool5_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool5_r2, [3, 3, 512, 128], 'pool5_conv_r2')), 0.5)
        pool5_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool5_conv_r2, [1, 1, 128, 128], 'pool5_fc_r2')), 0.5)
        pool5_ms_saliency_r2 = self.conv2d(pool5_fc_r2, [1, 1, 128, 1], 'pool5_ms_saliency_r2')

        pool4_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_r2, [3, 3, 512, 128], 'pool4_conv_r2')), 0.5)
        pool4_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv_r2, [1, 1, 128, 128], 'pool4_fc_r2')), 0.5)
        pool4_ms_saliency_r2 = self.conv2d(pool4_fc_r2, [1, 1, 128, 1], 'pool4_ms_saliency_r2')

        pool3_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_r2, [3, 3, 256, 128], 'pool3_conv_r2')), 0.5)
        pool3_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool3_conv_r2, [1, 1, 128, 128], 'pool3_fc_r2')), 0.5)
        pool3_ms_saliency_r2 = self.conv2d(pool3_fc_r2, [1, 1, 128, 1], 'pool3_ms_saliency_r2')

        pool2_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool2_r2, [3, 3, 128, 128], 'pool2_conv_r2')), 0.5)
        pool2_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool2_conv_r2, [1, 1, 128, 128], 'pool2_fc_r2')), 0.5)
        pool2_ms_saliency_r2 = self.conv2d(pool2_fc_r2, [1, 1, 128, 1], 'pool2_ms_saliency_r2')

        pool1_conv_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool1_r2, [3, 3, 64, 128], 'pool1_conv_r2')), 0.5)
        pool1_fc_r2 = tf.nn.dropout(tf.nn.relu(self.conv2d(pool1_conv_r2, [1, 1, 128, 128], 'pool1_fc_r2')), 0.5)
        pool1_ms_saliency_r2 = self.conv2d(pool1_fc_r2, [1, 1, 128, 1], 'pool1_ms_saliency_r2')

        ########## DSS structure ##########

        # fc8
        scale6_score = tf.image.resize_bilinear(fc8_r2, [size, size])

        scale5_score = self.conv2d(tf.concat([fc8_r2, pool5_ms_saliency_r2], axis=3), [1, 1, 2, 1], 'scale5')
        scale5_score = tf.image.resize_bilinear(scale5_score, [size, size])

        # fc8 + pool5 + pool4
        scale4_score = self.conv2d(tf.concat([fc8_r2, pool5_ms_saliency_r2, pool4_ms_saliency_r2], axis=3), [1, 1, 3, 1], 'scale4')
        scale4_score = tf.image.resize_bilinear(scale4_score, [size, size])

        # fc8 + pool5 + pool4 + pool3
        scale3_score = self.conv2d(tf.concat([fc8_r2, pool5_ms_saliency_r2, pool4_ms_saliency_r2, pool3_ms_saliency_r2], axis=3),
            [1, 1, 4, 1], 'scale3')
        scale3_score = tf.image.resize_bilinear(scale3_score, [size, size])

        # fc8 + pool5 + pool4 + pool3 + pool2
        pool2_size = pool2_ms_saliency_r2.get_shape().as_list()
        up2_fc8 = tf.image.resize_bilinear(fc8_r2, [pool2_size[1], pool2_size[2]])
        up2_pool5 = tf.image.resize_bilinear(pool5_ms_saliency_r2, [pool2_size[1], pool2_size[2]])
        up2_pool4 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [pool2_size[1], pool2_size[2]])
        up2_pool3 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [pool2_size[1], pool2_size[2]])
        scale2_score = self.conv2d(tf.concat([up2_fc8, up2_pool5, up2_pool4, up2_pool3, pool2_ms_saliency_r2], axis=3),
                                   [1, 1, 5, 1], 'scale2')
        scale2_score = tf.image.resize_bilinear(scale2_score, [size, size])

        # fc8 + pool5 + pool4 + pool3 + pool2 + pool1
        pool1_size = pool1_ms_saliency_r2.get_shape().as_list()
        up1_fc8 = tf.image.resize_bilinear(fc8_r2, [pool1_size[1], pool1_size[2]])
        up1_pool5 = tf.image.resize_bilinear(pool5_ms_saliency_r2, [pool1_size[1], pool1_size[2]])
        up1_pool4 = tf.image.resize_bilinear(pool4_ms_saliency_r2, [pool1_size[1], pool1_size[2]])
        up1_pool3 = tf.image.resize_bilinear(pool3_ms_saliency_r2, [pool1_size[1], pool1_size[2]])
        up1_pool2 = tf.image.resize_bilinear(pool2_ms_saliency_r2, [pool1_size[1], pool1_size[2]])
        scale1_score = self.conv2d(
            tf.concat([up1_fc8, up1_pool5, up1_pool4, up1_pool3, up1_pool2, pool1_ms_saliency_r2], axis=3),
            [1, 1, 6, 1], 'scale1')
        scale1_score = tf.image.resize_bilinear(scale1_score, [size, size])

        ########## rnn fusion ############

        inputs = tf.expand_dims(
            tf.concat([scale1_score, scale2_score, scale3_score, scale4_score, scale5_score, scale6_score], axis=3), 0)
        cell = ConvLSTMCell([512, 512], 6, [3, 3])
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, scope='rnn')
        rnn_output = tf.squeeze(outputs, axis=0)

        final_fusion = self.conv2d(rnn_output, [1, 1, 6, 1], 'final_saliency')

        self.final_saliency = tf.sigmoid(final_fusion)
        self.saver = tf.train.Saver()

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_fusion, labels=self.Y), name='loss')
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

    def train_ST_rnn(self, train_dir, label_dir, prior_dir, list_file_path):

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
        # save_path = 'tempImages'
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_loss = []
        log_loss.append('pre-trained model: dcl \n')

        for itr in range(50001):
            x, y = dataset.next_batch()
            # feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y}
            feed_dict = {self.X: x, self.Y: y}

            self.sess.run(self.train_op, feed_dict=feed_dict)

            if itr % 5 == 0:
                train_loss, saliency, summary_str = self.sess.run([self.loss, self.final_saliency, summary_op], feed_dict=feed_dict)
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
            # batch_x_no_prior = np.zeros([4, 512, 512, 3])
            for i, image_name in enumerate(images_path):
                image = Image.open(os.path.join(test_dir, image_name + '.jpg'))
                prior = Image.open(os.path.join(test_prior_dir, image_name + '.png'))
                image, prior = resize_image_prior(image, prior)
                w, h = image.size
                input_prior, input = preprocess(image, prior)

                # batch_x_no_prior[i] = input
                batch_x[i] = input_prior


            feed_dict = {self.X: batch_x}

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
            save_sal = saliency.astype(np.uint8)
            save_img = Image.fromarray(save_sal[3, :h, :w, 0])

            image_path = os.path.join(save_path, images_path[-1] + '.png')
            print ('process:', image_path)
            print('time:', end - start)
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




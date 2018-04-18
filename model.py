import tensorflow as tf
import numpy as np
import math
from PIL import Image
from cell import ConvLSTMCell
import cv2
from matplotlib import pyplot as plt
from image_data_loader import ImageData, ImageAndPriorData, ImageAndPriorSeqData
import os
import time
from utils import preprocess
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

    def __init__(self, sess, batch_size, lr=0.00001, ckpt_dir='./parameters'):
        self.sess = sess
        self.ckpt_dir = ckpt_dir
        self.lr = lr
        self.batch_size = batch_size

        self.build_ST_RNN()
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
        # rnn_output_fc8 = self.rnn_cell(fc8, 'fc8')
        # fc8 = tf.add(rnn_output_fc8, fc8)

        up_fc8 = tf.image.resize_bilinear(fc8, [512, 512])

        pool4_conv = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4, [3, 3, 512, 128], 'pool4_conv')), 0.5)
        pool4_fc = tf.nn.dropout(tf.nn.relu(self.conv2d(pool4_conv, [1, 1, 128, 128], 'pool4_fc')), 0.5)
        pool4_ms_saliency = self.conv2d(pool4_fc, [1, 1, 128, 1], 'pool4_ms_saliency')

        # rnn_output_pool4 = self.rnn_cell(pool4_ms_saliency, 'pool4')
        # pool4_ms_saliency = tf.add(rnn_output_pool4, pool4_ms_saliency)

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

        ########## rnn fusion ############

        inputs = tf.expand_dims(tf.concat([up_pool4, up_pool4_r2, up_fc8, up_fc8_r2], axis=3), 0)
        cell = ConvLSTMCell([512, 512], 1, [3, 3])
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, scope='rnn')
        rnn_output = tf.squeeze(outputs, axis=0)

        ########### ST fusion ############
        pool4_saliency_cancat = tf.concat([pool4_ms_saliency, pool4_ms_saliency_r2], 3, name='concat_pool4')
        pool4_saliency_ST = self.conv2d(pool4_saliency_cancat, [1, 1, 2, 1], 'pool4_saliency_ST')
        up_pool4_ST = tf.image.resize_bilinear(pool4_saliency_ST, [512, 512])

        fc8_concat = tf.concat([fc8, fc8_r2], 3, name='concat_fc8')
        fc8_saliency_ST = self.conv2d(fc8_concat, [1, 1, 2, 1], 'fc8_saliency_ST')
        up_fc8_ST = tf.image.resize_bilinear(fc8_saliency_ST, [512, 512])

        final_saliency = tf.add(up_pool4_ST, up_fc8_ST)
        # final_saliency = tf.add(final_saliency, up_pool4_r2)
        # final_saliency = tf.add(final_saliency, up_fc8_r2)

        final_saliency = tf.add(final_saliency, rnn_output)
        # final_saliency = tf.add(final_saliency, attention_dynamic)
        self.final_saliency = tf.sigmoid(final_saliency)
        self.up_fc8 = up_fc8
        self.rnn_output = final_saliency
        self.saver = tf.train.Saver()

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=self.Y),
                                   name='loss')
        self.loss_rnn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rnn_output, labels=self.Y),
                                   name='loss2')
        tf.summary.scalar('entropy', self.loss + 0.1 * self.loss_rnn)
        # tf.summary.scalar('entropy', self.loss)
        optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
        trainable_var = tf.trainable_variables()
        grads = optimizer.compute_gradients(self.loss + 0.1 * self.loss_rnn, var_list=trainable_var)
        # grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
        self.train_op = optimizer.apply_gradients(grads)


    def train_ST_rnn(self, train_dir, label_dir, prior_dir, list_file_path):

        list_file = open(list_file_path)
        image_names = [line.strip() for line in list_file]


        # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
        dataset = ImageAndPriorSeqData(train_dir, label_dir, prior_dir, None, None,
                                       None,
                                       image_names, None, '.jpg', '.png', 550, 512, 1, 4,
                                       horizontal_flip=False)

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)

        # save_path = 'tempImages'
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


        for itr in xrange(26001):
            x, y = dataset.next_batch()
            feed_dict = {self.X: x[:, :, :, :3], self.X_prior: x, self.Y: y}
            # self.sess.run([self.loss, self.final_saliency, self.up_fc8, self.rnn_output, summary_op],
            #     feed_dict=feed_dict)

            self.sess.run(self.train_op, feed_dict=feed_dict)

            if itr % 5 == 0:
                train_loss, saliency, rnn_output, summary_str = self.sess.run(
                    [self.loss, self.final_saliency, self.rnn_output, summary_op],
                    feed_dict=feed_dict)
                print 'step: %d, train_loss:%g' % (itr, train_loss)
                summary_writer.add_summary(summary_str, itr)

            if itr % 2000 == 0:
                self.save(str(itr), time_str)



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
                image, prior = resize_image_prior(image, prior)
                w, h = image.size
                input_prior, input = preprocess(image, prior)

                batch_x_no_prior[i] = input
                batch_x[i] = input_prior

            feed_dict = {self.X: batch_x_no_prior, self.X_prior: batch_x}
            saliency = self.sess.run(self.final_saliency, feed_dict=feed_dict)

            saliency = saliency * 255
            save_sal = saliency.astype(np.uint8)
            save_img = Image.fromarray(save_sal[3, :h, :w, 0])

            image_path = os.path.join(save_path, images_path[-1] + '.png')
            print 'process:', image_path
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

    def save(self, itr, network_name):
        model_dir = os.path.join('models', network_name, itr)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        print 'save model:', os.path.join(model_dir, 'snap_model.ckpt')
        self.saver.save(self.sess, os.path.join(model_dir, 'snap_model.ckpt'))


    def load(self):
        self.sess.run(self.init)
        self.saver.restore(self.sess, self.ckpt_dir)




    def forward(self, image):
        # tf.initialize_all_variables().run()
        # self.sampler(image)
        return self.sess.run(self.final_saliency, feed_dict = {self.X_test: image})




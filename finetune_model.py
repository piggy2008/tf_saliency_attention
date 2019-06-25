import tensorflow as tf
import h5py
import numpy as np
from utils import preprocess
from PIL import Image
import os
from model_regreession import resize_image_prior
import matplotlib.pyplot as plt
from image_data_loader import ImageAndPriorSeqData
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scipy.io as sio

parameter_path = 'models/2018-07-01 11:02:01/24000/snap_model.ckpt'
# def readParameterName():
#     reader = tf.train.NewCheckpointReader(parameter_path)
#
#     for key in sorted(reader.get_variable_to_shape_map()):
#         print(key)
#         print(reader.get_tensor(key).shape)
#
# readParameterName()

parameter_path = 'models/2018-07-01 11:02:01/24000/snap_model.ckpt.meta'

def finetune_model():
    image_dir = '/home/ty/data/video_saliency/train_all'
    label_dir = '/home/ty/data/video_saliency/train_all_gt2_revised'
    prior_dir = '/home/ty/data/video_saliency/train_all_prior'
    list_file_path = '/home/ty/data/video_saliency/train_all_seq.txt'

    list_file = open(list_file_path)
    image_names = [line.strip() for line in list_file]

    # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
    dataset = ImageAndPriorSeqData(image_dir, label_dir, prior_dir, None, None,
                                   None,
                                   image_names, None, '.jpg', '.png', 550, 512, 1,
                                   4,
                                   horizontal_flip=False)

    learning_rate = 0.0001
    saver = tf.train.import_meta_graph(parameter_path)

    with tf.Session() as sess:

        graph = tf.get_default_graph()
        final_saliency = graph.get_tensor_by_name('Add_3:0')
        Y_holder = graph.get_tensor_by_name('gt:0')
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=Y_holder),
                              name='classify_loss')
        optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer_new')
        trainable_var = tf.trainable_variables()
        grads = optimizer.compute_gradients(loss, var_list=trainable_var)
        train_op = optimizer.apply_gradients(grads)

        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, 'models/2018-07-01 11:02:01/24000/snap_model.ckpt')

        # attention_conv1_w = graph.get_tensor_by_name('pool4_saliency_ST_w:0')
        # print (sess.run(attention_conv1_w))
        for itr in range(16001):
            x, y = dataset.next_batch()
            feed_dict = {'rgb_image:0': x[:, :, :, :3], 'rgb_prior_image:0': x, 'gt:0': y}

            sess.run(train_op, feed_dict=feed_dict)

def extrac_parameter():

    saver = tf.train.import_meta_graph(parameter_path)

    with tf.Session() as sess:

        graph = tf.get_default_graph()
        # final_saliency = graph.get_tensor_by_name('Add_3:0')
        # Y_holder = graph.get_tensor_by_name('gt:0')
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_saliency, labels=Y_holder),
        #                       name='classify_loss')
        #
        save_mat = {}
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, 'models/2018-07-01 11:02:01/24000/snap_model.ckpt')

        conv1_1_w = graph.get_tensor_by_name('conv1_1_w:0')
        conv1_2_w = graph.get_tensor_by_name('conv1_2_w:0')
        conv1_1_b = graph.get_tensor_by_name('conv1_1_b:0')
        conv1_2_b = graph.get_tensor_by_name('conv1_2_b:0')

        save_mat['conv1_1_w'] = conv1_1_w.eval()
        save_mat['conv1_2_w'] = conv1_2_w.eval()
        save_mat['conv1_1_b'] = conv1_1_b.eval()
        save_mat['conv1_2_b'] = conv1_2_b.eval()

        conv2_1_w = graph.get_tensor_by_name('conv2_1_w:0')
        conv2_2_w = graph.get_tensor_by_name('conv2_2_w:0')
        conv2_1_b = graph.get_tensor_by_name('conv2_1_b:0')
        conv2_2_b = graph.get_tensor_by_name('conv2_2_b:0')

        save_mat['conv2_1_w'] = conv2_1_w.eval()
        save_mat['conv2_2_w'] = conv2_2_w.eval()
        save_mat['conv2_1_b'] = conv2_1_b.eval()
        save_mat['conv2_2_b'] = conv2_2_b.eval()

        conv3_1_w = graph.get_tensor_by_name('conv3_1_w:0')
        conv3_2_w = graph.get_tensor_by_name('conv3_2_w:0')
        conv3_3_w = graph.get_tensor_by_name('conv3_3_w:0')
        conv3_1_b = graph.get_tensor_by_name('conv3_1_b:0')
        conv3_2_b = graph.get_tensor_by_name('conv3_2_b:0')
        conv3_3_b = graph.get_tensor_by_name('conv3_3_b:0')

        save_mat['conv3_1_w'] = conv3_1_w.eval()
        save_mat['conv3_2_w'] = conv3_2_w.eval()
        save_mat['conv3_3_w'] = conv3_3_w.eval()
        save_mat['conv3_1_b'] = conv3_1_b.eval()
        save_mat['conv3_2_b'] = conv3_2_b.eval()
        save_mat['conv3_3_b'] = conv3_3_b.eval()

        conv4_1_w = graph.get_tensor_by_name('conv4_1_w:0')
        conv4_2_w = graph.get_tensor_by_name('conv4_2_w:0')
        conv4_3_w = graph.get_tensor_by_name('conv4_3_w:0')
        conv4_1_b = graph.get_tensor_by_name('conv4_1_b:0')
        conv4_2_b = graph.get_tensor_by_name('conv4_2_b:0')
        conv4_3_b = graph.get_tensor_by_name('conv4_3_b:0')

        save_mat['conv4_1_w'] = conv4_1_w.eval()
        save_mat['conv4_2_w'] = conv4_2_w.eval()
        save_mat['conv4_3_w'] = conv4_3_w.eval()
        save_mat['conv4_1_b'] = conv4_1_b.eval()
        save_mat['conv4_2_b'] = conv4_2_b.eval()
        save_mat['conv4_3_b'] = conv4_3_b.eval()

        conv5_1_w = graph.get_tensor_by_name('conv5_1_w:0')
        conv5_2_w = graph.get_tensor_by_name('conv5_2_w:0')
        conv5_3_w = graph.get_tensor_by_name('conv5_3_w:0')
        conv5_1_b = graph.get_tensor_by_name('conv5_1_b:0')
        conv5_2_b = graph.get_tensor_by_name('conv5_2_b:0')
        conv5_3_b = graph.get_tensor_by_name('conv5_3_b:0')

        save_mat['conv5_1_w'] = conv5_1_w.eval()
        save_mat['conv5_2_w'] = conv5_2_w.eval()
        save_mat['conv5_3_w'] = conv5_3_w.eval()
        save_mat['conv5_1_b'] = conv5_1_b.eval()
        save_mat['conv5_2_b'] = conv5_2_b.eval()
        save_mat['conv5_3_b'] = conv5_3_b.eval()

        fc6_w = graph.get_tensor_by_name('fc6_w:0')
        fc7_w = graph.get_tensor_by_name('fc7_w:0')
        fc8_w = graph.get_tensor_by_name('fc8_w:0')
        fc6_b = graph.get_tensor_by_name('fc6_b:0')
        fc7_b = graph.get_tensor_by_name('fc7_b:0')
        fc8_b = graph.get_tensor_by_name('fc8_b:0')

        save_mat['fc6_w'] = fc6_w.eval()
        save_mat['fc7_w'] = fc7_w.eval()
        save_mat['fc8_w'] = fc8_w.eval()
        save_mat['fc6_b'] = fc6_b.eval()
        save_mat['fc7_b'] = fc7_b.eval()
        save_mat['fc8_b'] = fc8_b.eval()

        pool3_conv_w = graph.get_tensor_by_name('pool3_conv_w:0')
        pool3_fc_w = graph.get_tensor_by_name('pool3_fc_w:0')
        pool3_ms_saliency_w = graph.get_tensor_by_name('pool3_ms_saliency_w:0')
        pool3_conv_b = graph.get_tensor_by_name('pool3_conv_b:0')
        pool3_fc_b = graph.get_tensor_by_name('pool3_fc_b:0')
        pool3_ms_saliency_b = graph.get_tensor_by_name('pool3_ms_saliency_b:0')

        save_mat['pool3_conv_w'] = pool3_conv_w.eval()
        save_mat['pool3_fc_w'] = pool3_fc_w.eval()
        save_mat['pool3_ms_saliency_w'] = pool3_ms_saliency_w.eval()
        save_mat['pool3_conv_b'] = pool3_conv_b.eval()
        save_mat['pool3_fc_b'] = pool3_fc_b.eval()
        save_mat['pool3_ms_saliency_b'] = pool3_ms_saliency_b.eval()

        pool4_conv_w = graph.get_tensor_by_name('pool4_conv_w:0')
        pool4_fc_w = graph.get_tensor_by_name('pool4_fc_w:0')
        pool4_ms_saliency_w = graph.get_tensor_by_name('pool4_ms_saliency_w:0')
        pool4_conv_b = graph.get_tensor_by_name('pool4_conv_b:0')
        pool4_fc_b = graph.get_tensor_by_name('pool4_fc_b:0')
        pool4_ms_saliency_b = graph.get_tensor_by_name('pool4_ms_saliency_b:0')

        save_mat['pool4_conv_w'] = pool4_conv_w.eval()
        save_mat['pool4_fc_w'] = pool4_fc_w.eval()
        save_mat['pool4_ms_saliency_w'] = pool4_ms_saliency_w.eval()
        save_mat['pool4_conv_b'] = pool4_conv_b.eval()
        save_mat['pool4_fc_b'] = pool4_fc_b.eval()
        save_mat['pool4_ms_saliency_b'] = pool4_ms_saliency_b.eval()

        #################################################
        conv1_1_r2_w = graph.get_tensor_by_name('conv1_1_r2_w:0')
        conv1_2_r2_w = graph.get_tensor_by_name('conv1_2_r2_w:0')
        conv1_1_r2_b = graph.get_tensor_by_name('conv1_1_r2_b:0')
        conv1_2_r2_b = graph.get_tensor_by_name('conv1_2_r2_b:0')

        save_mat['conv1_1_r2_w'] = conv1_1_r2_w.eval()
        save_mat['conv1_2_r2_w'] = conv1_2_r2_w.eval()
        save_mat['conv1_1_r2_b'] = conv1_1_r2_b.eval()
        save_mat['conv1_2_r2_b'] = conv1_2_r2_b.eval()

        conv2_1_r2_w = graph.get_tensor_by_name('conv2_1_r2_w:0')
        conv2_2_r2_w = graph.get_tensor_by_name('conv2_2_r2_w:0')
        conv2_1_r2_b = graph.get_tensor_by_name('conv2_1_r2_b:0')
        conv2_2_r2_b = graph.get_tensor_by_name('conv2_2_r2_b:0')

        save_mat['conv2_1_r2_w'] = conv2_1_r2_w.eval()
        save_mat['conv2_2_r2_w'] = conv2_2_r2_w.eval()
        save_mat['conv2_1_r2_b'] = conv2_1_r2_b.eval()
        save_mat['conv2_2_r2_b'] = conv2_2_r2_b.eval()

        conv3_1_r2_w = graph.get_tensor_by_name('conv3_1_r2_w:0')
        conv3_2_r2_w = graph.get_tensor_by_name('conv3_2_r2_w:0')
        conv3_3_r2_w = graph.get_tensor_by_name('conv3_3_r2_w:0')
        conv3_1_r2_b = graph.get_tensor_by_name('conv3_1_r2_b:0')
        conv3_2_r2_b = graph.get_tensor_by_name('conv3_2_r2_b:0')
        conv3_3_r2_b = graph.get_tensor_by_name('conv3_3_r2_b:0')

        save_mat['conv3_1_r2_w'] = conv3_1_r2_w.eval()
        save_mat['conv3_2_r2_w'] = conv3_2_r2_w.eval()
        save_mat['conv3_3_r2_w'] = conv3_3_r2_w.eval()
        save_mat['conv3_1_r2_b'] = conv3_1_r2_b.eval()
        save_mat['conv3_2_r2_b'] = conv3_2_r2_b.eval()
        save_mat['conv3_3_r2_b'] = conv3_3_r2_b.eval()

        conv4_1_r2_w = graph.get_tensor_by_name('conv4_1_r2_w:0')
        conv4_2_r2_w = graph.get_tensor_by_name('conv4_2_r2_w:0')
        conv4_3_r2_w = graph.get_tensor_by_name('conv4_3_r2_w:0')
        conv4_1_r2_b = graph.get_tensor_by_name('conv4_1_r2_b:0')
        conv4_2_r2_b = graph.get_tensor_by_name('conv4_2_r2_b:0')
        conv4_3_r2_b = graph.get_tensor_by_name('conv4_3_r2_b:0')

        save_mat['conv4_1_r2_w'] = conv4_1_r2_w.eval()
        save_mat['conv4_2_r2_w'] = conv4_2_r2_w.eval()
        save_mat['conv4_3_r2_w'] = conv4_3_r2_w.eval()
        save_mat['conv4_1_r2_b'] = conv4_1_r2_b.eval()
        save_mat['conv4_2_r2_b'] = conv4_2_r2_b.eval()
        save_mat['conv4_3_r2_b'] = conv4_3_r2_b.eval()

        conv5_1_r2_w = graph.get_tensor_by_name('conv5_1_r2_w:0')
        conv5_2_r2_w = graph.get_tensor_by_name('conv5_2_r2_w:0')
        conv5_3_r2_w = graph.get_tensor_by_name('conv5_3_r2_w:0')
        conv5_1_r2_b = graph.get_tensor_by_name('conv5_1_r2_b:0')
        conv5_2_r2_b = graph.get_tensor_by_name('conv5_2_r2_b:0')
        conv5_3_r2_b = graph.get_tensor_by_name('conv5_3_r2_b:0')

        save_mat['conv5_1_r2_w'] = conv5_1_r2_w.eval()
        save_mat['conv5_2_r2_w'] = conv5_2_r2_w.eval()
        save_mat['conv5_3_r2_w'] = conv5_3_r2_w.eval()
        save_mat['conv5_1_r2_b'] = conv5_1_r2_b.eval()
        save_mat['conv5_2_r2_b'] = conv5_2_r2_b.eval()
        save_mat['conv5_3_r2_b'] = conv5_3_r2_b.eval()

        fc6_r2_w = graph.get_tensor_by_name('fc6_r2_w:0')
        fc7_r2_w = graph.get_tensor_by_name('fc7_r2_w:0')
        fc8_r2_w = graph.get_tensor_by_name('fc8_r2_w:0')
        fc6_r2_b = graph.get_tensor_by_name('fc6_r2_b:0')
        fc7_r2_b = graph.get_tensor_by_name('fc7_r2_b:0')
        fc8_r2_b = graph.get_tensor_by_name('fc8_r2_b:0')

        save_mat['fc6_r2_w'] = fc6_r2_w.eval()
        save_mat['fc7_r2_w'] = fc7_r2_w.eval()
        save_mat['fc8_r2_w'] = fc8_r2_w.eval()
        save_mat['fc6_r2_b'] = fc6_r2_b.eval()
        save_mat['fc7_r2_b'] = fc7_r2_b.eval()
        save_mat['fc8_r2_b'] = fc8_r2_b.eval()

        pool3_conv_r2_w = graph.get_tensor_by_name('pool3_conv_r2_w:0')
        pool3_fc_r2_w = graph.get_tensor_by_name('pool3_fc_r2_w:0')
        pool3_ms_saliency_r2_w = graph.get_tensor_by_name('pool3_ms_saliency_r2_w:0')
        pool3_conv_r2_b = graph.get_tensor_by_name('pool3_conv_r2_b:0')
        pool3_fc_r2_b = graph.get_tensor_by_name('pool3_fc_r2_b:0')
        pool3_ms_saliency_r2_b = graph.get_tensor_by_name('pool3_ms_saliency_r2_b:0')

        save_mat['pool3_conv_r2_w'] = pool3_conv_r2_w.eval()
        save_mat['pool3_fc_r2_w'] = pool3_fc_r2_w.eval()
        save_mat['pool3_ms_saliency_r2_w'] = pool3_ms_saliency_r2_w.eval()
        save_mat['pool3_conv_r2_b'] = pool3_conv_r2_b.eval()
        save_mat['pool3_fc_r2_b'] = pool3_fc_r2_b.eval()
        save_mat['pool3_ms_saliency_r2_b'] = pool3_ms_saliency_r2_b.eval()

        pool4_conv_r2_w = graph.get_tensor_by_name('pool4_conv_r2_w:0')
        pool4_fc_r2_w = graph.get_tensor_by_name('pool4_fc_r2_w:0')
        pool4_ms_saliency_r2_w = graph.get_tensor_by_name('pool4_ms_saliency_r2_w:0')
        pool4_conv_r2_b = graph.get_tensor_by_name('pool4_conv_r2_b:0')
        pool4_fc_r2_b = graph.get_tensor_by_name('pool4_fc_r2_b:0')
        pool4_ms_saliency_r2_b = graph.get_tensor_by_name('pool4_ms_saliency_r2_b:0')

        save_mat['pool4_conv_r2_w'] = pool4_conv_r2_w.eval()
        save_mat['pool4_fc_r2_w'] = pool4_fc_r2_w.eval()
        save_mat['pool4_ms_saliency_r2_w'] = pool4_ms_saliency_r2_w.eval()
        save_mat['pool4_conv_r2_b'] = pool4_conv_r2_b.eval()
        save_mat['pool4_fc_r2_b'] = pool4_fc_r2_b.eval()
        save_mat['pool4_ms_saliency_r2_b'] = pool4_ms_saliency_r2_b.eval()

        sio.savemat('mat_parameter/pretrained_attention.mat', save_mat)






def test_model():
    list_file_path = '/home/ty/data/FBMS/FBMS_seq_file.txt'
    test_dir = '/home/ty/data/FBMS/FBMS_Testset'
    test_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
    list_file = open(list_file_path)
    test_names = [line.strip() for line in list_file]

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(parameter_path)
        saver.restore(sess, 'models/best/snap_model.ckpt')

        graph = tf.get_default_graph()
        attention_conv1_w = graph.get_tensor_by_name('pool4_saliency_ST_w:0')
        print (sess.run(attention_conv1_w))

        for test_path in test_names:

            images_path = test_path.split(',')
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

            feed_dict = {'rgb_image:0': batch_x_no_prior, 'rgb_prior_image:0': batch_x}
            saliency = graph.get_tensor_by_name('Sigmoid:0')
            result = sess.run(saliency, feed_dict=feed_dict)
            plt.subplot(2, 1, 1)
            plt.imshow(result[3, :h, :w, 0])

            plt.subplot(2, 1, 2)
            plt.imshow(result[3, :h, :w, 0])

            plt.show()

if __name__ == '__main__':
    extrac_parameter()
    # test_model()

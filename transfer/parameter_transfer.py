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
parameter = sio.loadmat('DCL_parameter.mat')

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

print conv1_1_b.shape
conv1_1_w = tf.Variable(np.transpose(conv1_1_w, [2, 3, 1, 0]))
conv1_2_w = tf.Variable(np.transpose(conv1_2_w, [2, 3, 1, 0]))
conv2_1_w = tf.Variable(np.transpose(conv2_1_w, [2, 3, 1, 0]))
conv2_2_w = tf.Variable(np.transpose(conv2_2_w, [2, 3, 1, 0]))
conv3_1_w = tf.Variable(np.transpose(conv3_1_w, [2, 3, 1, 0]))
conv3_2_w = tf.Variable(np.transpose(conv3_2_w, [2, 3, 1, 0]))
conv3_3_w = tf.Variable(np.transpose(conv3_3_w, [2, 3, 1, 0]))
conv4_1_w = tf.Variable(np.transpose(conv4_1_w, [2, 3, 1, 0]))
conv4_2_w = tf.Variable(np.transpose(conv4_2_w, [2, 3, 1, 0]))
conv4_3_w = tf.Variable(np.transpose(conv4_3_w, [2, 3, 1, 0]))

conv5_1_w = tf.Variable(np.transpose(conv5_1_w, [2, 3, 1, 0]))
conv5_2_w = tf.Variable(np.transpose(conv5_2_w, [2, 3, 1, 0]))
conv5_3_w = tf.Variable(np.transpose(conv5_3_w, [2, 3, 1, 0]))

fc6_w = tf.Variable(np.transpose(fc6_w, [2, 3, 1, 0]))
fc7_w = tf.Variable(np.transpose(fc7_w, [2, 3, 1, 0]))
fc8_w = tf.Variable(np.transpose(fc8_w, [2, 3, 1, 0]))

pool4_conv_w = tf.Variable(np.transpose(pool4_conv_w, [2, 3, 1, 0]))
pool4_fc_w = tf.Variable(np.transpose(pool4_fc_w, [2, 3, 1, 0]))
pool4_ms_saliency_w = tf.Variable(np.transpose(pool4_ms_saliency_w, [2, 3, 1, 0]))
# conv1_1_w = conv1_1_w[:, 0, :, :]
# conv1_1_w = conv1_1_w[:, :, :, np.newaxis]
x = tf.placeholder(tf.float32, [None, 512, 512, 3])
# x_image = tf.reshape(x, [-1, 28, 28, 1])

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
final_saliency = tf.sigmoid(tf.add(up_pool4, up_fc8))
print final_saliency

saver = tf.train.Saver()
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

    im = Image.open('/home/ty/0_0_899.jpg')
    w, h = im.size
    if max(w, h) > 510:
        if w > h:
            im = im.resize([510, int(510. / w * h)])

        else:
            im = im.resize([int(510. / h * w), 510])


    w, h = im.size
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]


    mean = (104.00699, 116.66877, 122.67892)
    mean = np.array(mean)

    in_ = (in_ - mean)
    npad = ([0, 512 - h], [0, 512 - w], [0, 0])
    in_ = np.pad(in_, npad, 'constant')
    in_ = in_[np.newaxis, ...]
    print np.shape(in_)
    feed_dict = {x: in_}
    result = sess.run(final_saliency, feed_dict)
    print result
    print np.shape(result)
    plt.imshow(result[0,:,:,0])
    plt.show()
    # saver.save(sess, 'DCL_tensorflow.ckpt')

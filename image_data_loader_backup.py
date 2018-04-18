import numpy as np
import tensorflow as tf
import os
from PIL import Image
from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use('GTKAgg')
import random
import cv2


class ImageData(object):
    def __init__(self, image_dir, label_dir, validate_image_dir, validate_label_dir, image_suffix, label_suffix,
                 image_size, crop_size, batch_size, horizontal_flip=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.image_size = image_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.batch_size = batch_size
        self.validate_image_dir = validate_image_dir
        self.validate_label_dir = validate_label_dir
        self._load_image_name()
        self._reset_batch_offset()

    def _load_image_name(self):
        temp_names = os.listdir(self.image_dir)
        self.num_of_image = len(temp_names)
        temp_names.sort()
        print temp_names
        full_names = []
        for full_name in temp_names:
            name, suffix = os.path.splitext(full_name)
            full_names.append(name)

        self.image_name = full_names

    def _crop_image(self, x, y, random_crop_size, sync_seed=None):
        np.random.seed(sync_seed)

        h, w = x.shape[0], x.shape[1]
        rangeh = (h - random_crop_size[0]) // 2
        rangew = (w - random_crop_size[1]) // 2
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)

        h_start, h_end = offseth, offseth + random_crop_size[0]
        w_start, w_end = offsetw, offsetw + random_crop_size[1]

        return x[h_start:h_end, w_start:w_end, :], y[h_start:h_end, w_start:w_end, :]

    def _flip_image(self, x, y):
        axis = 1
        if np.random.random() < 0.5:

            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)

            y = np.asarray(y).swapaxes(axis, 0)
            y = y[::-1, ...]
            y = y.swapaxes(0, axis)
        return x, y

    def _preprocess(self, x):
        x = x[:, :, ::-1]
        mean = (104.00699, 116.66877, 122.67892)
        mean = np.array(mean, dtype=np.float32)
        x = (x - mean)
        return x

    def _reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self):
        start = self.batch_offset
        self.batch_offset += self.batch_size
        if self.batch_offset > self.num_of_image:
            random.shuffle(self.image_name)
            start = 0
            self.batch_offset = self.batch_size

        end = self.batch_offset
        if self.crop_size:
            batch_x = np.zeros([self.batch_size, self.crop_size, self.crop_size, 3])
            batch_y = np.zeros([self.batch_size, self.crop_size, self.crop_size, 1])
        else:
            batch_x = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
            batch_y = np.zeros([self.batch_size, self.image_size, self.image_size, 1])
        count = 0
        for index in range(start, end):
            image_path = os.path.join(self.image_dir, self.image_name[index] + self.image_suffix)
            label_path = os.path.join(self.label_dir, self.image_name[index] + self.label_suffix)
            image = Image.open(image_path)
            label = Image.open(label_path)
            image = image.resize([self.image_size, self.image_size])
            label = label.resize([self.image_size, self.image_size])
            x = np.array(image, dtype=np.float32)
            y = np.array(label, dtype=np.uint8)
            y = y.reshape((y.shape[0], y.shape[1], 1))
            if self.crop_size:
                x, y = self._crop_image(x, y, (self.crop_size, self.crop_size))
            if self.horizontal_flip:
                x, y = self._flip_image(x, y)
            x = self._preprocess(x)
            batch_x[count] = x
            batch_y[count] = y
            count += 1

        return batch_x, batch_y

    def get_validate_images(self):
        if self.validate_image_dir and self.validate_label_dir:
            images = os.listdir(self.validate_image_dir)
            batch_x = np.zeros([len(images), self.crop_size, self.crop_size, 3])
            batch_y = np.zeros([len(images), self.crop_size, self.crop_size, 1])
            count = 0
            for image in images:
                name, suffix = os.path.splitext(image)
                image_path = os.path.join(self.validate_image_dir, name + self.image_suffix)
                label_path = os.path.join(self.validate_label_dir, name + self.label_suffix)

                image = Image.open(image_path)
                label = Image.open(label_path)
                image = image.resize([self.crop_size, self.crop_size])
                label = label.resize([self.crop_size, self.crop_size])
                x = np.array(image, dtype=np.float32)
                y = np.array(label, dtype=np.int8)
                y = y.reshape((y.shape[0], y.shape[1], 1))

                x = self._preprocess(x)
                batch_x[count] = x
                batch_y[count] = y
                count += 1

            return batch_x, batch_y


class ImageAndPriorData(ImageData):
    def __init__(self, image_dir, label_dir, prior_dir, validate_image_dir, validate_label_dir, validate_prior_dir,
                 image_names, validate_names, image_suffix, label_suffix,
                 image_size, crop_size, batch_size, horizontal_flip=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.prior_dir = prior_dir
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.image_size = image_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.batch_size = batch_size
        self.validate_image_dir = validate_image_dir
        self.validate_label_dir = validate_label_dir
        self.validate_prior_dir = validate_prior_dir
        self.image_names = image_names
        self.validate_names = validate_names
        self._load_image_name()
        self._reset_batch_offset()
        # random.shuffle(self.image_names)

    def _load_image_name(self):
        self.num_of_image = len(self.image_names)

    def next_batch(self):
        start = self.batch_offset
        self.batch_offset += self.batch_size
        if self.batch_offset > self.num_of_image:
            random.shuffle(self.image_names)
            start = 0
            self.batch_offset = self.batch_size

        end = self.batch_offset
        # print start, '---', end
        if self.crop_size:
            batch_x = np.zeros([self.batch_size, self.crop_size, self.crop_size, 4])
            batch_y = np.zeros([self.batch_size, self.crop_size, self.crop_size, 1])
        else:
            batch_x = np.zeros([self.batch_size, self.image_size, self.image_size, 4])
            batch_y = np.zeros([self.batch_size, self.image_size, self.image_size, 1])
        count = 0
        for index in range(start, end):
            image_path = os.path.join(self.image_dir, self.image_names[index] + self.image_suffix)
            label_path = os.path.join(self.label_dir, self.image_names[index] + self.label_suffix)
            prior_path = os.path.join(self.prior_dir, self.image_names[index] + self.label_suffix)

            # print label_path

            image = cv2.imread(image_path)
            label = cv2.imread(label_path, 0)
            prior = cv2.imread(prior_path, 0)
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            prior = cv2.resize(prior, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

            x = image.astype(dtype=np.float32)
            x = self._preprocess(x)
            prior_arr = prior.astype(dtype=np.float32)
            input = np.zeros([self.image_size, self.image_size, 4], dtype=np.float32)
            input[:, :, :3] = x
            input[:, :, 3] = prior_arr
            y = label.astype(dtype=np.float32)
            y /= 255
            y = y.astype(np.uint8)
            y = y.reshape((y.shape[0], y.shape[1], 1))
            if self.crop_size:
                input, y = self._crop_image(input, y, (self.crop_size, self.crop_size))
            if self.horizontal_flip:
                input, y = self._flip_image(input, y)

            batch_x[count] = input
            batch_y[count] = y
            count += 1

        return batch_x, batch_y

    def get_validate_images(self):
        if self.validate_image_dir and self.validate_label_dir:
            image_size = len(self.validate_names)
            batch_x = np.zeros([image_size, self.crop_size, self.crop_size, 4])
            batch_y = np.zeros([image_size, self.crop_size, self.crop_size, 1])
            count = 0
            for image_name in self.validate_names:

                image_path = os.path.join(self.validate_image_dir, image_name + self.image_suffix)
                label_path = os.path.join(self.validate_label_dir, image_name + self.label_suffix)
                prior_path = os.path.join(self.validate_prior_dir, image_name + self.label_suffix)

                image = Image.open(image_path)
                label = Image.open(label_path)
                prior = Image.open(prior_path)

                image = image.resize([self.crop_size, self.crop_size])
                label = label.resize([self.crop_size, self.crop_size])
                prior = prior.resize([self.crop_size, self.crop_size])

                x = np.array(image, dtype=np.float32)
                x = self._preprocess(x)
                prior_arr = np.array(prior, dtype=np.float32)
                input = np.zeros([self.crop_size, self.crop_size, 4], dtype=np.float32)
                input[:, :, :3] = x
                input[:, :, 3] = prior_arr

                y = np.array(label, dtype=np.float32)
                y /= 255
                y = y.reshape((y.shape[0], y.shape[1], 1))

                batch_x[count] = input
                batch_y[count] = y
                count += 1

            return batch_x, batch_y

class ImageAndPriorSeqData(ImageData):
    def __init__(self, image_dir, label_dir, prior_dir, validate_image_dir, validate_label_dir, validate_prior_dir,
                 image_names, validate_names, image_suffix, label_suffix,
                 image_size, crop_size, batch_size, seq_size, horizontal_flip=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.prior_dir = prior_dir
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.image_size = image_size
        self.crop_size = crop_size
        self.seq_size = seq_size
        self.horizontal_flip = horizontal_flip
        self.batch_size = batch_size
        self.validate_image_dir = validate_image_dir
        self.validate_label_dir = validate_label_dir
        self.validate_prior_dir = validate_prior_dir
        self.image_names = image_names
        self.validate_names = validate_names
        self._load_image_name()
        self._reset_batch_offset()
        random.shuffle(self.image_names)

    def _load_image_name(self):
        self.num_of_image = len(self.image_names)

    def next_batch(self):
        start = self.batch_offset
        self.batch_offset += self.batch_size
        if self.batch_offset > self.num_of_image:
            random.shuffle(self.image_names)
            start = 0
            self.batch_offset = self.batch_size

        end = self.batch_offset
        # print start, '---', end
        if self.crop_size:
            batch_x = np.zeros([self.seq_size, self.crop_size, self.crop_size, 4])
            batch_y = np.zeros([self.seq_size, self.crop_size, self.crop_size, 1])
        else:
            batch_x = np.zeros([self.seq_size, self.image_size, self.image_size, 4])
            batch_y = np.zeros([self.seq_size, self.image_size, self.image_size, 1])
        count = 0
        for index in range(start, end):
            # print label_path
            images_path = self.image_names[index].split(',')

            for image_name in images_path:
                image_path = os.path.join(self.image_dir, image_name + self.image_suffix)
                label_path = os.path.join(self.label_dir, image_name + self.label_suffix)
                prior_path = os.path.join(self.prior_dir, image_name + self.label_suffix)

                image = cv2.imread(image_path)
                label = cv2.imread(label_path, 0)
                prior = cv2.imread(prior_path, 0)

                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                prior = cv2.resize(prior, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

                x = image.astype(dtype=np.float32)
                x = self._preprocess(x)
                prior_arr = prior.astype(dtype=np.float32)
                input = np.zeros([self.image_size, self.image_size, 4], dtype=np.float32)
                input[:, :, :3] = x
                input[:, :, 3] = prior_arr
                y = label.astype(dtype=np.float32)
                y /= 255
                y = y.astype(np.uint8)
                y = y.reshape((y.shape[0], y.shape[1], 1))
                if self.crop_size:
                    input, y = self._crop_image(input, y, (self.crop_size, self.crop_size))
                if self.horizontal_flip:
                    input, y = self._flip_image(input, y)

                batch_x[count] = input
                batch_y[count] = y
                count += 1

        return batch_x, batch_y

    def get_validate_images(self):
        if self.validate_image_dir and self.validate_label_dir:
            image_size = len(self.validate_names)
            batch_x = np.zeros([image_size, self.crop_size, self.crop_size, 4])
            batch_y = np.zeros([image_size, self.crop_size, self.crop_size, 1])
            count = 0
            for image_name in self.validate_names:

                image_path = os.path.join(self.validate_image_dir, image_name + self.image_suffix)
                label_path = os.path.join(self.validate_label_dir, image_name + self.label_suffix)
                prior_path = os.path.join(self.validate_prior_dir, image_name + self.label_suffix)

                image = Image.open(image_path)
                label = Image.open(label_path)
                prior = Image.open(prior_path)

                image = image.resize([self.crop_size, self.crop_size])
                label = label.resize([self.crop_size, self.crop_size])
                prior = prior.resize([self.crop_size, self.crop_size])

                x = np.array(image, dtype=np.float32)
                x = self._preprocess(x)
                prior_arr = np.array(prior, dtype=np.float32)
                input = np.zeros([self.crop_size, self.crop_size, 4], dtype=np.float32)
                input[:, :, :3] = x
                input[:, :, 3] = prior_arr

                y = np.array(label, dtype=np.float32)
                y /= 255
                y = y.reshape((y.shape[0], y.shape[1], 1))

                batch_x[count] = input
                batch_y[count] = y
                count += 1

            return batch_x, batch_y



if __name__ == '__main__':

    image_dir = '/home/ty/data/davis/480p'
    label_dir = '/home/ty/data/davis/GT'
    prior_dir = '/home/ty/data/davis/davis_flow_prior'
    davis_file = open('/home/ty/data/davis/davis_seq_file.txt')
    image_names = [line.strip() for line in davis_file]

    validate_dir = '/home/ty/data/FBMS/FBMS_Testset2'
    validate_label_dir = '/home/ty/data/FBMS/GT_no_first'
    validate_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
    FBMS_file = open('/home/ty/data/FBMS/FBMS_file.txt')
    validate_names = [line.strip() for line in FBMS_file]
    # dataset = ImageData(image_dir, label_dir, '.jpg', '.png', 550, 512, 1, horizontal_flip=True)
    dataset = ImageAndPriorSeqData(image_dir, label_dir, prior_dir, validate_dir, validate_label_dir, validate_prior_dir,
                                image_names, validate_names, '.jpg', '.png', 550, 512, 1, 4, horizontal_flip=False)

    x, y = dataset.next_batch()

    # validate_x, validate_y = dataset.get_validate_images()
    # for i in range(0, 10000):
    #     x, y = dataset.next_batch()
    #     print i
    print x.shape
    print y.shape
    # print np.unique(y)
    # plt.subplot(1, 3, 1)
    # plt.imshow(x[0, :, :, :3].astype(np.uint8))
    # plt.subplot(1, 3, 2)
    # plt.imshow(x[0, :, :, 3])
    # plt.subplot(1, 3, 3)
    # plt.imshow(y[0, :, :, 0])
    # plt.show()
    # x = x.astype(dtype=np.uint8)

    # img = Image.fromarray(x[0, :, :, :], mode='P')

    # print result
    # print np.shape(result)
    # with tf.Session() as sess:
    #     dcl = DCL(sess, 'parameters/DCL_tensorflow.ckpt')
    #     dcl.sampler(x)
    #     dcl.load()
    #     result = dcl.forward(x)
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(x[0, :, :, :])
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(result[0, :, :, 0])
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(y[0, :, :, 0])
    #     plt.show()

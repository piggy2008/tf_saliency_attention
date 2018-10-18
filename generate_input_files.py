import os
from PIL import Image
import cv2
import numpy as np

path = '/home/ty/data/video_saliency/train_all_gt2_no_coarse'
save_path = '/home/ty/data/video_saliency/train_no_coarse_seq.txt'
# path = '/home/ty/data/davis/davis_test'
# save_path = '/home/ty/data/davis/davis_test_seq.txt'
# save_path = '/home/ty/data/video_saliency/train_all_seq.txt'
folders = os.listdir(path)
file = open(save_path, 'w')

batch = 4

def generate_one():

    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            name, suffix = os.path.splitext(image)
            print (os.path.join(folder, name))
            file.writelines(os.path.join(folder, name) + '\n')

    file.close()

def generate_seq():
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for i in range(1, len(images) - batch + 1):
            image_batch = ''
            for j in range(batch):

                image = images[i + j]
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 3:
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print (image_batch)
            file.writelines(image_batch + '\n')

    file.close()

def change_suffix():
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            img = Image.open(os.path.join(path, folder, image))
            name, suffix = os.path.splitext(image)
            if not os.path.exists(os.path.join(save_path, folder)):
                os.makedirs(os.path.join(save_path, folder))
            img.save(os.path.join(save_path, folder, name + '.jpg'))

def gt_generate():
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            img = cv2.imread(os.path.join(path, folder, image), 0)
            img[np.where(img > 100)] = 255
            img[np.where(img <= 100)] = 0
            # img[np.where(img == 255)] = 1
            if not os.path.exists(os.path.join(save_path, folder)):
                os.makedirs(os.path.join(save_path, folder))

            cv2.imwrite(os.path.join(save_path, folder, image), img)

# generate_one()

generate_seq()
# change_suffix()
# generate_seq()
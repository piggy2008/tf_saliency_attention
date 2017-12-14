import os

path = '/home/ty/data/davis/davis_flow_prior'
save_path = '/home/ty/data/davis/davis_seq_file.txt'
folders = os.listdir(path)
file = open(save_path, 'w')

batch = 4

def generate_one():

    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for image in images:
            name, suffix = os.path.splitext(image)
            print os.path.join(folder, name)
            file.writelines(os.path.join(folder, name) + '\n')

    file.close()

def generate_seq():
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        images.sort()
        for i in range(0, len(images) - batch + 1):
            image_batch = ''
            for j in xrange(batch):

                image = images[i + j]
                name, suffix = os.path.splitext(image)
                path_temp = os.path.join(folder, name)
                if j == 3:
                    image_batch = image_batch + path_temp
                else:
                    image_batch = image_batch + path_temp + ','
            print image_batch
            file.writelines(image_batch + '\n')

    file.close()

generate_seq()
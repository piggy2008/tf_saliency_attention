import os

path = '/home/ty/data/FBMS/GT_no_first'
save_path = '/home/ty/data/FBMS/FBMS_file.txt'
folders = os.listdir(path)
file = open(save_path, 'w')

for folder in folders:
    images = os.listdir(os.path.join(path, folder))
    images.sort()
    for image in images:
        name, suffix = os.path.splitext(image)
        print os.path.join(folder, name)
        file.writelines(os.path.join(folder, name) + '\n')

file.close()
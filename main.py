from model import VideoSailency
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    with tf.Session() as sess:

        parameter_path = 'models/best/snap_model.ckpt' #### best
        parameter_path = 'models/2018-04-15 11:29:15/14000/snap_model.ckpt'
        # parameter_path = 'fusion_parameter_backup/fusionST_tensorflow.ckpt'
        # test dir
        test_dir = '/home/ty/data/FBMS/FBMS_Testset'
        test_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
        list_file_path ='/home/ty/data/FBMS/FBMS_seq_file.txt'
        save_path = 'total_result/result_rnn'


        vs = VideoSailency(sess, 4, ckpt_dir=parameter_path)
        #
        vs.test_seq(test_dir, test_prior_dir, list_file_path, save_path)


        # train dir
        # image_dir = '/home/ty/data/video_saliency/train_all'
        # label_dir = '/home/ty/data/video_saliency/train_all_gt2'
        # prior_dir = '/home/ty/data/video_saliency/train_all_prior'
        # list_file_path = '/home/ty/data/video_saliency/train_no_coarse_seq.txt'
        # vs.train_ST_rnn(image_dir, label_dir, prior_dir, list_file_path)
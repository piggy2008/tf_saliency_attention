from model_dss import VideoSailency
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        phrase = 'test'
        prior_type = 'prior'
        drop_path = True
        if phrase == 'train':

            # parameter_path = 'fusion_C3D_ms_parameter/fusionST_C3D_ms_tensorflow.ckpt'
            # parameter_path = 'models/2018-05-24 17:01:51/6000/snap_model.ckpt'  #2018.5.25
            # parameter_path = 'models/2018-05-30 21:11:24/6000/snap_model.ckpt'
            parameter_path = 'fusion_dss_structure_parameter/fusionST_LSTM_DSS.ckpt'
            vs = VideoSailency(sess, 4, image_size=530, crop_size=512, lr=0.0001, ckpt_dir=parameter_path)
            image_dir = '/home/ty/data/video_saliency/train_all'
            label_dir = '/home/ty/data/video_saliency/train_all_gt2_revised'
            prior_dir = '/home/ty/data/video_saliency/train_all_prior'
            list_file_path = '/home/ty/data/video_saliency/train_all_seq.txt'
            vs.train_ST_rnn(image_dir, label_dir, prior_dir, list_file_path)
        else:
            # parameter_path = 'models/2018-05-30 09:33:19/2000/snap_model.ckpt' #3d path total best result
            # parameter_path = 'models/2018-05-29 20:11:18/6000/snap_model.ckpt'
            # parameter_path = 'models/2018-07-01 11:02:01/24000/snap_model.ckpt' #3d path total attention best result
            parameter_path = 'models/2018-11-03 22:19:18/50000/snap_model.ckpt'
            # # test dir
            # FBMS
            # test_dir = '/home/ty/data/FBMS/FBMS_Testset'
            # test_prior_dir = '/home/ty/data/FBMS/FBMS_Testset_flow_prior'
            # list_file_path = '/home/ty/data/FBMS/FBMS_seq_file.txt'
            # DAVIS
            test_dir = '/home/ty/data/davis/davis_test'
            test_prior_dir = '/home/ty/data/davis/davis_flow_prior'
            list_file_path = '/home/ty/data/davis/davis_test_seq.txt'

            save_path = 'total_result/result_rnn'
            vs = VideoSailency(sess, 4, ckpt_dir=parameter_path)
            #
            vs.test_seq(test_dir, test_prior_dir, list_file_path, save_path)



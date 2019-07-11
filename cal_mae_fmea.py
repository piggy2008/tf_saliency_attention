import os
import numpy as np
from PIL import Image
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure

# root = '/home/qub/data/saliency/davis/davis_test2'
root_inference = '/home/ty/code/tf_saliency_attention/total_result/result_rnn_2018-08-04 11:08:00/'
root = '/home/ty/data/davis/480p/'
name = 'davis'
# gt_root = '/home/qub/data/saliency/davis/GT'
gt_root = '/home/ty/data/davis/GT/'
# gt_root = '/home/qub/data/saliency/VOS/GT'

precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
results = {}

# save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']))
folders = os.listdir(root_inference)
folders.sort()
for folder in folders:
    imgs = os.listdir(os.path.join(root_inference, folder))
    imgs.sort()

    for img in imgs:
        print(os.path.join(folder, img))
        image = Image.open(os.path.join(root, folder, img[:-4] + '.jpg')).convert('RGB')
        gt = np.array(Image.open(os.path.join(gt_root, folder, img)).convert('L'))
        pred = np.array(Image.open(os.path.join(root_inference, folder, img)).convert('L'))

        precision, recall, mae = cal_precision_recall_mae(pred, gt)

        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        mae_record.update(mae)

fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

print ('test results:')
print (results)

# THUR15K + DAVIS snap:10000 {'davis': {'mae': 0.03617724595417807, 'fmeasure': 0.8150494537915058}}
# THUR15K + DAVIS(input no mea & std) snap:30000 {'davis': {'mae': 0.03403602471853535}, 'fmeasure': 0.8208723312824877}
# THUR15K + DAVIS snap:30000 {'davis': {'mae': 0.02795341027164935}, 'fmeasure': 0.846696146351338}
# THUR15K + DAVIS resize:473*473 snap:30000 {'davis': 'mae': 0.02464488739008121, ''fmeasure': 0.8753527027151914}
# THUR15K + DAVIS resize:473*473 model:R1 high and low, snap:30000 {'davis': {'fmeasure': 0.8657611483587979, 'mae': 0.028688147260396805}}
# THUR15K + DAVIS resize:473*473 model: model prior recurrent snap:30000 {'davis': {'mae': 0.02533309706615563, 'fmeasure': 0.8745875295714605}}
# THUR15K + DAVIS resize:473*473 model: model prior recurrent + feature maps plus
# snap:30000 {'davis': {'fmeasure': 0.8751256401745396, 'mae': 0.025352599605078505}}

# VideoSaliency_2019-05-03 00:54:21 is better, using model_prior, R3Net base and add previous frame supervision and recurrent GRU motion extraction
# training details, first, directly train R3Net using DAFB2 and THUR15K, second, finetune the model by add recurrent module and GRU, then finetune twice
# using DAFB2 and THUR15K but dataloader shuffle=false in order to have consecutive frames. The specific super parameter is in VideoSaliency_2019-05-03 00:54:21
# VideoSaliency_2019-05-01 23:29:39 and VideoSaliency_2019-04-20 23:11:17/30000.pth

# VideoSaliency_2019-05-03 23:59:44: finetune model prior from 05-01 model, fix other layers excepet motion module
# {'davis': {'mae': 0.031455319655690664, 'fmeasure': 0.8687384596915435}}

# VideoSaliency_2019-05-14 17:13:16: no finetune
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# {'davis': {'fmeasure': 0.8760938218680382, 'mae': 0.03375186721061853}}

# VideoSaliency_2019-05-15 03:06:29: finetune model prior from VideoSaliency_2019-05-14 17:13:16 model, train entire network with lr:1e-6
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# {'davis': {'fmeasure': 0.8770158996877871, 'mae': 0.03235241246303723}}

# VideoSaliency_2019-05-15 03:06:29: finetune model prior from VideoSaliency_2019-05-14 17:13:16 model, train entire network with lr:1e-5
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# {'davis': {'mae': 0.02977316776424702, 'fmeasure': 0.8773961688318479}}
# {'FBMS': {'fmeasure': 0.8462238927200698, 'mae': 0.05929029351096353}}

# VideoSaliency_2019-05-17 03:27:37: finetune model prior from VideoSaliency_2019-05-14 17:13:16 model, train entire network with lr:1e-5
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# model: self-attention + motion enhancement + prior attention weight learning
# {'FBMS': {'fmeasure': 0.8431560452294077, 'mae': 0.0572594186609631}}
# {'FBMS': {'mae': 0.05151967407911611, 'fmeasure': 0.8512965990283861}} with crf
# {'VOS': {'fmeasure': 0.7693856907104227, 'mae': 0.07323270547216723}}
# {'VOS': {'mae': 0.061354405913717075, 'fmeasure': 0.76979294074132}} with crf
# {'SegTrackV2': {'fmeasure': 0.8900102827035228, 'mae': 0.02371825726384187}}
# {'SegTrackV2': {'mae': 0.01414643253248216, 'fmeasure': 0.8974274867145704}} with CRF
# {'MCL': {'fmeasure': 0.7941665988086701, 'mae': 0.03365593652205517}}
# {'MCL': {'fmeasure': 0.8033409666446579, 'mae': 0.030916401685247424}} with crf
# {'ViSal': {'mae': 0.01547489956096272, 'fmeasure': 0.9517413442552852}}
# {'ViSal': {'fmeasure': 0.9541724935997185, 'mae': 0.009944043273381801}} with crf
# {'davis': {'fmeasure': 0.877271448077333, 'mae': 0.028900763530552247}}
# {'davis': {'fmeasure': 0.8877485369547635, 'mae': 0.017803576387589698}} with crf

# VideoSaliency_2019-06-25 00:58:16 traning from original resnext50 model of torch parameter, using dataset:DUT-TR + DAVIS model: raw R3Net lr:0.001
# {'davis': {'fmeasure': 0.8697159794201897, 'mae': 0.035606949365716525}}

# VideoSaliency_2019-06-25 17:44:55 traning from original resnext50 model of torch parameter, using dataset:DUT-TR + DAVIS
# finetune VideoSaliency_2019-06-25 00:58:16 model:resnext50 + R3Net + GRU + motion enhancement + saliency guide block
# {'davis': {'mae': 0.034840332588290626, 'fmeasure': 0.8566140865851933}}

# VideoSaliency_2019-06-25 00:42:59 traning from original resnext50 model of torch parameter, using dataset:DUT-OMRON + DAVIS model: raw R3Net lr:0.001
# {'davis': {'mae': 0.026136525479663834, 'fmeasure': 0.8583683681098009}}

# VideoSaliency_2019-06-25 18:35:28 traning from original resnext50 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-25 00:42:59 model:resnext50 + R3Net + GRU + motion enhancement + saliency guide block
# {'davis': {'mae': 0.027190812067633865, 'fmeasure': 0.8564467742823199}}

# VideoSaliency_2019-06-25 18:46:13 traning from original resnext50 model of torch parameter, using dataset:DUT-OMRON + DAVIS model: raw R3Net lr:0.001
# no self_attention
# {'davis': {'mae': 0.028268766387925158, 'fmeasure': 0.8641904514712092}}

# VideoSaliency_2019-06-26 00:07:16 traning from original resnext101 model of torch parameter, using dataset:DUT-TR + DAVIS model: raw R3Net lr:0.001
# no self_attention
# {'davis': {'fmeasure': 0.8744918145377412, 'mae': 0.028497783782586317}}

# VideoSaliency_2019-06-26 00:49:01 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS model: raw R3Net lr:0.001
# no self_attention
# {'davis': {'mae': 0.02956942893893325, 'fmeasure': 0.8636986541096229}}

# VideoSaliency_2019-06-26 18:08:11(20000)traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16 model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block
# no self_attention
# {'davis': {'mae': 0.027365099548091857, 'fmeasure': 0.8843037688863674}} 20000
# {'davis': {'fmeasure': 0.882500630734551, 'mae': 0.028656513632573044}} 30000
# {'MCL': {'fmeasure': 0.7797077599010649, 'mae': 0.03490434030025704}}
# {'ViSal': {'mae': 0.015330959018609812, 'fmeasure': 0.949898057517949}}
# {'FBMS': {'mae': 0.06180595295896911, 'fmeasure': 0.8339872466494525}}
# {'VOS': {'mae': 0.07404737148285567, 'fmeasure': 0.759238636002531}}

# VideoSaliency_2019-06-26 18:42:54 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:49:01 model:resnext101 + R3Net + GRU + motion enhancement + saliency guide block
# no self_attention
# {'davis': {'fmeasure': 0.8632752266559531, 'mae': 0.029003498754963063}} 20000
# {'davis': {'fmeasure': 0.864925, 'mae': 0.0289121}} 30000


# VideoSaliency_2019-07-11 00:09:51 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16 model:resnext101 + R3Net + no motion block + motion enhancement + saliency guide block
# no self_attention
# {'davis': {'fmeasure': 0.88136, 'mae': 0.028495}} 30000
# {'davis': {'fmeasure': 0.884773, 'mae': 0.0271906}} 20000
# {'davis': {'fmeasure': 0.885063, 'mae': 0.0285210}} 10000
# {'FBMS': {'mae': 0.05814119064871477, 'fmeasure': 0.840094490963166}}
# {'MCL': {'mae': 0.0345719343627967, 'fmeasure': 0.7831879684122651}}

# VideoSaliency_2019-07-11 00:09:51 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16 model:resnext101 + R3Net + motion block + motion enhancement + saliency guide block
# no self_attention motion block channel is 256 to 64
# {'davis': {'mae': 0.027200999676079675, 'fmeasure': 0.8819144797935291}}
# {'FBMS': {'mae': 0.058138874315261733, 'fmeasure': 0.8362034796659599}}
# {'ViSal': {'mae': 0.01453721749511041, 'fmeasure': 0.9498225723351809}}
# {'MCL': {'mae': 0.033866961735088005, 'fmeasure': 0.7857094621253102}}
# {'SegTrackV2': {'mae': 0.023640045394179604, 'fmeasure': 0.8774014546489907}}
# {'VOS': {'mae': 0.07308082925115635, 'fmeasure': 0.761926607316529}}
# {'UVSD': {'mae': 0.03607475861206765, 'fmeasure': 0.7011512131755825}}

# VideoSaliency_2019-06-27 00:56:18 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-26 00:07:16 model:resnext101 + R3Net + no motion block + motion enhancement
# no self_attention
# {'davis':  {'mae': 0.027359359945784614, 'fmeasure': 0.8840887753328227}} 30000
# {'davis': {'mae': 0.017455241696254332, 'fmeasure': 0.8905493155021433}} with crf


# VideoSaliency_2019-06-28 22:24:22 traning from original resnet50 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-28 18:33:02 model:resnet50 + R3Net + GRU + motion enhancement + saliency guide block
# no self_attention
# {'MCL': {'fmeasure': 0.7759971653989058, 'mae': 0.033103970530375657}}
# {'davis': {'mae': 0.02952947523541565, 'fmeasure': 0.8601778871578505}}

# VideoSaliency_2019-06-27 19:00:52 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-05-14 17:13:16 model:resnext101 + R3Net + GRU + motion enhancement + no saliency guide block
# have self_attention
# {'davis': {'fmeasure': 0.8751047167822349, 'mae': 0.03059594151109138}} 20000
# {'FBMS': {'mae': 0.05827975689206127, 'fmeasure': 0.8395263230105977}} 20000
# {'MCL': {'fmeasure': 0.7897838103776252, 'mae': 0.0341193568105776}} 20000

# VideoSaliency_2019-07-04 00:05:22 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-05-14 17:13:16 model:resnext101 + R3Net + no motion block + motion enhancement + no saliency guide block
# have self_attention
# {'davis': {'fmeasure': 0.874, 'mae': 0.0306}} 20000
# {'FBMS': {'mae': 0.841, 'fmeasure': 0.0573}} 20000
# {'MCL': {'fmeasure': 0.7978824584120152, 'mae': 0.03270654914596075}} 20000

# {'FBMS': {'mae': 0.05898730165783062, 'fmeasure': 0.8394230953578746}} 10000

# VideoSaliency_2019-07-01 22:14:16 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-05-14 17:13:16 model:resnext101 + R3Net + no motion block + motion enhancement
# have self_attention
# {'davis': {'mae': 0.030584487861882274, 'fmeasure': 0.8761062884262997}}
# {'FBMS': {'mae': 0.05905227985899478, 'fmeasure': 0.8397254002559016}}
# {'MCL': {'fmeasure': 0.7899358517201228, 'mae': 0.033703267332709085}}

# VideoSaliency_2019-07-01 17:32:33 traning from original resnet101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-06-28 22:46:18 model:resnet101 + R3Net + GRU + motion enhancement + saliency guide block
# no self_attention
# {'davis': {'fmeasure': 0.8502764930743375, 'mae': 0.026479752498128697}}

# VideoSaliency_2019-07-02 04:18:40 traning from original resnet101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-07-01 23:50:14 model:resnet101 + R3Net + GRU + motion enhancement + saliency guide block
# have self_attention
# {'davis': {'mae': 0.02984727148613444, 'fmeasure': 0.8756064480635604}}
# {'MCL': {'fmeasure': 0.7657679645629271, 'mae': 0.033812008388706564}}
# {'FBMS': {'mae': 0.05899833031167441, 'fmeasure': 0.829591156478791}}

# VideoSaliency_2019-07-02 04:10:31 traning from original resnet50 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-07-01 23:51:39 model:resnet50 + R3Net + GRU + motion enhancement + saliency guide block
# have self_attention
# {'davis': {'fmeasure': 0.8572470140650937, 'mae': 0.03127575596579786}}
# {'FBMS': {'mae': 0.065907850117285, 'fmeasure': 0.8169151015700682}}
# {'MCL': {'fmeasure': 0.7650784843213123, 'mae': 0.03801076001268533}}


# VideoSaliency_2019-07-02 17:47:38 traning from original resnext50 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-07-02 04:21:40 model:resnext50 + R3Net + GRU + motion enhancement + saliency guide block
# have self_attention
# {'davis': {'fmeasure': 0.867985179773024, 'mae': 0.031125308718449575}}
# {'FBMS': {'mae': 0.05804739949561263, 'fmeasure': 0.8342736409382326}}
# {'MCL': {'fmeasure': 0.7773786755646357, 'mae': 0.034323540436367886}}

# VideoSaliency_2019-07-02 17:43:09: no finetune, only resnext101
# using dataset:DUT-TR + DAVIS R3Net pre-train
# {'davis': {'fmeasure': 0.8653577036343185, 'mae': 0.04190723255244401}} 20000
# {'davis': {'mae': 0.04510454890929087, 'fmeasure': 0.8482913046852011}} 15000
# {'FBMS': {'mae': 0.060115473676380815, 'fmeasure': 0.8370974160080146}} 15000
# {'MCL': {'fmeasure': 0.7719638752799074, 'mae': 0.03398585222840852}} 20000
# {'MCL': {'fmeasure': 0.7643802009424475, 'mae': 0.037740759818777427}} 15000
# {'SegTrackV2': {'fmeasure': 0.8833902740630123, 'mae': 0.019613184126835628}} 20000
# {'SegTrackV2': {'fmeasure': 0.8661164437880786, 'mae': 0.022897514345448376}} 15000
# {'VOS': {'fmeasure': 0.7689440285547244, 'mae': 0.06967526071368603}} 20000

# VideoSaliency_2019-07-02 23:41:10: no finetune, only resnext101 no self_attention
# using dataset:DUT-TR + DAVIS R3Net pre-train
# {'FBMS': {'fmeasure': 0.8408206546850922, 'mae': 0.05546505643440736}} 30000
# {'MCL': {'mae': 0.03155394830915426, 'fmeasure': 0.7538068242918796}} 30000

# VideoSaliency_2019-07-02 23:42:38, only resnext101 no self_attention
# using dataset:DUT-OMRON + DAVIS R3Net pre-train
# {'davis': {'fmeasure': 0.8566587089872602, 'mae': 0.027955859338707836}} 30000
# {'davis': {'mae': 0.03151518507370385, 'fmeasure': 0.8296323204360312}} 20000
# {'FBMS': {'fmeasure': 0.8275514791821478, 'mae': 0.05934688602250718}} 30000
# {'FBMS': {'fmeasure': 0.8191922301416684, 'mae': 0.062295793605747364}}

# VideoSaliency_2019-07-03 17:36:37 traning from original resnext101 model of torch parameter, using dataset:DUT-OMRON + DAVIS
# finetune VideoSaliency_2019-05-14 17:13:16 model:resnext101 + R3Net + LSTM + motion enhancement + saliency guide block
# {'MCL': {'fmeasure': 0.7919051602536508, 'mae': 0.033465254438458325}}
# {'davis': {'fmeasure': 0.8739062844885623, 'mae': 0.03299489011624521}}
# {'FBMS': {'fmeasure': 0.8381842471006478, 'mae': 0.058207687328754}}


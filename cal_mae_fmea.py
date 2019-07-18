import os
import numpy as np
from PIL import Image
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure

# root = '/home/qub/data/saliency/davis/davis_test2'
# root_inference = '/home/qub/data/davis/others/SCNN'
# root = '/home/qub/data/saliency/davis/480p/'

root_inference = '/home/qub/code/tf_saliency_attention/results/result_rnn_crf_2019-07-17 22:14:51'
root = '/home/qub/data/saliency/FBMS/FBMS_Testset2/'

# root = '/home/qub/data/saliency/FBMS/FBMS_Testset/'
name = 'STC_FBMS'
# gt_root = '/home/qub/data/saliency/davis/GT'
gt_root = '/home/qub/data/saliency/FBMS/GT_no_first'
# gt_root = '/home/qub/data/saliency/FBMS/GT/'


precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
results = {}

# save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']))

# folders =['bear','blackswan','bmx-bumps','bmx-trees','boat','breakdance','breakdance-flare','bus',
#               'camel','car-roundabout','car-shadow','car-turn','cows','dance-jump','dance-twirl','dog',
#               'dog-agility','drift-chicane','drift-straight','drift-turn','elephant','flamingo','goat','hike',
#               'hockey','horsejump-high','horsejump-low','kite-surf','kite-walk','libby','lucia','mallard-fly',
#               'mallard-water','motocross-bumps','motocross-jump','motorbike','paragliding','paragliding-launch',
#               'parkour','rhino','rollerblade','scooter-black','scooter-gray','soapbox','soccerball','stroller','surf','swing',
#               'tennis','train']
# folders = ['blackswan','bmx-trees','breakdance', 'camel','car-roundabout','car-shadow','cows','dance-twirl','dog',
#            'drift-chicane','drift-straight','goat',
#              'horsejump-high','kite-surf','libby',
#               'motocross-jump','paragliding-launch',
#               'parkour','scooter-black','soapbox']

# folders = ['horsejump-high','horsejump-low','kite-surf','kite-walk','libby','lucia','mallard-fly',
#    'mallard-water','motocross-bumps','motocross-jump','motorbike','paragliding','paragliding-launch',
#    'parkour','rhino','rollerblade','scooter-black','scooter-gray','soapbox','soccerball','stroller','surf','swing',
#    'tennis','train']
folders = os.listdir(root)
folders.sort()
for folder in folders:
    imgs = os.listdir(os.path.join(root_inference, folder))
    imgs.sort()
    # imgs = imgs[:-1]

    for img in imgs:
        # if img == 'tennis463.png' or img == 'tennis473.png' or img == 'tennis483.png'or img == 'tennis490.png':
        #     continue
        print(os.path.join(folder, img))
        image = Image.open(os.path.join(root, folder, img[:-4] + '.jpg')).convert('RGB')
        pred = Image.open(os.path.join(root_inference, folder, img[:-4] + '.png')).convert('L')
        gt = Image.open(os.path.join(gt_root, folder, img[:-4] + '.png')).convert('L')
        gt = gt.resize(pred.size)
        image = image.resize(pred.size)
        gt = np.array(gt)
        pred = np.array(pred)

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

# {'davis': {'mae': 0.041576569176772944, 'fmeasure': 0.8341383096984007}}
# {'MSST_davis': {'fmeasure': 0.8175943834081874, 'mae': 0.04597473876855389}}
# {'Amulet_davis': {'mae': 0.08374974551689243, 'fmeasure': 0.7234079968968813}}
# {'CG_davis': {'fmeasure': 0.6278087775523111, 'mae': 0.09568971798828023}}
# {'CS_davis': {'fmeasure': 0.387371123540425, 'mae': 0.11592338609834756}}
# {'DCL_davis': {'fmeasure': 0.7555328232313439, 'mae': 0.1325773794024856}}
# {'DHSN_davis': {'fmeasure': 0.7850133247155778, 'mae': 0.0403306345651108}}
# {'DSMT_davis': {'fmeasure': 0.7341665103336742, 'mae': 0.08759877182362823}}
# {'DSS_davis': {'mae': 0.06691170192462226, 'fmeasure': 0.7481973606361545}}
# {'mdf_davis': {'fmeasure': 0.6840797691135195, 'mae': 0.10272304482305192}}
# {'rfcn_davis': {'fmeasure': 0.7319531392937486, 'mae': 0.06884022953284744}}
# {'SA_davis': {'fmeasure': 0.5281544937902222, 'mae': 0.1064141104007172}}
# {'ST_daivs': {'fmeasure': 0.5197071620892447, 'mae': 0.14053913940188467}}
# {'SCNN_DAVIS_davis': {'mae': 0.07487107977582136, 'fmeasure': 0.7747607685667066}}
# {'SF_davis': {'mae': 0.09569977093725415, 'fmeasure': 0.2942410040011359}}
# {'SFCN_davis': {'fmeasure': 0.7496333741687372, 'mae': 0.055637025273703795}}
# {'SS_davis': {'mae': 0.4235285375857071, 'fmeasure': 0.37639028057715057}}
# {'UCF_davis': {'fmeasure': 0.7411008408000632, 'mae': 0.10889593152795193}}
# {'WSS_davis': {'mae': 0.07330220691994715, 'fmeasure': 0.6755352024596707}}
# {'PDB_davis': {'fmeasure': 0.8625471839500165, 'mae': 0.02943239025479776}}

# {'ours_FBMS': {'mae': 0.08512593678113392, 'fmeasure': 0.8126898801024736}}
# {'MSST_FBMS': {'mae': 0.0868737797152738, 'fmeasure': 0.7979339523459194}}
# {'Amulet_FBMS': {'mae': 0.11058736603477708, 'fmeasure': 0.7473186485664933}}
# {'CG_FBMS': {'fmeasure': 0.6060423808431531, 'mae': 0.17209793795519282}}
# {'CS_FBMS': {'mae': 0.17656805714883772, 'fmeasure': 0.42640692018722226}}
# {'DCL_FBMS': {'fmeasure': 0.7740927821309755, 'mae': 0.15355765996179924}}
# {'DHSN_FBMS': {'mae': 0.08660498628546363, 'fmeasure': 0.7608617035013118}}
# {'DSMT_FBMS': {'fmeasure': 0.7270150856031858, 'mae': 0.12363689728456925}}
# {'DSS_FBMS': {'mae': 0.08216943567138703, 'fmeasure': 0.7889026183064255}}
# {'mdf_FBMS': {'fmeasure': 0.6749914620952312, 'mae': 0.1347784315926681}}
# {'rfcn_FBMS': {'fmeasure': 0.7571037758902048, 'mae': 0.10839098950203489}}
# {'SA_FBMS': {'fmeasure': 0.5679453093482038, 'mae': 0.1823480449183362}}
# {'ST_FBMS': {'fmeasure': 0.5818536063862867, 'mae': 0.18009324457604398}}
# {'SCNN_FBMS': {'mae': 0.10564177043468248, 'fmeasure': 0.7809413356141277}}
# {'SF_FBMS': {'fmeasure': 0.33394517325171647, 'mae': 0.18458560094953252}}
# {'SFCN_FBMS': {'mae': 0.1028566042707361, 'fmeasure': 0.7561436193971052}}
# {'SS_FBMS': {'fmeasure': 0.2923778576371739, 'mae': 0.37033080923824857}}
# {'UCF_FBMS': {'mae': 0.1503423448920881, 'fmeasure': 0.7167000157282485}}
# {'WSS_FBMS': {'fmeasure': 0.7073782129195212, 'mae': 0.1172874031371225}}
# {'PDB_FBMS': {'fmeasure': 0.8267123735411919, 'mae': 0.07334640650355166}}



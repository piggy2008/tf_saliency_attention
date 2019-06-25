import scipy.io as sio

old = sio.loadmat('../mat_parameter/fusionST_parameter.mat')
addition = sio.loadmat('../mat_parameter/DCL_parameter.mat')
# sio.savemat(dcl, {'new': np.zeros([5, 5])})
# print dcl.keys()

param = {}

for key in old.keys():
    param[key] = old[key]


param['pool3_conv_w'] = addition['pool3_conv_w']
param['pool3_fc_w'] = addition['pool3_fc_w']
param['pool3_ms_saliency_w'] = addition['pool3_ms_saliency_w']

param['pool3_conv_b'] = addition['pool3_conv_b']
param['pool3_fc_b'] = addition['pool3_fc_b']
param['pool3_ms_saliency_b'] = addition['pool3_ms_saliency_b']

param['pool3_conv_r2_w'] = addition['pool3_conv_w']
param['pool3_fc_r2_w'] = addition['pool3_fc_w']
param['pool3_ms_saliency_r2_w'] = addition['pool3_ms_saliency_w']

param['pool3_conv_r2_b'] = addition['pool3_conv_b']
param['pool3_fc_r2_b'] = addition['pool3_fc_b']
param['pool3_ms_saliency_r2_b'] = addition['pool3_ms_saliency_b']

sio.savemat('../mat_parameter/fusionST_parameter_ms.mat', param)
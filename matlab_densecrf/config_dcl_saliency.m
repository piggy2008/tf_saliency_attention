function config = config_dcl_saliency()

%% CRF refinement
config.CRF.CRF_DIR = 'densecrf/prog_refine_pascal_v4';
config.CRF.im_sz = 514;
config.CRF.CRF_ITER = 3;
config.CRF.px = 3;
config.CRF.py = 3;
config.CRF.pw = 3;
config.CRF.bx = 50;
config.CRF.by = 50;
config.CRF.br = 3;
config.CRF.bg = 3;
config.CRF.bb = 3;
config.CRF.bw = 5;
%%

end
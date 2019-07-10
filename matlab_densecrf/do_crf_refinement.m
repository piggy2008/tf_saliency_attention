function dcl_crf =  do_crf_refinement(config, imgfilename, img, inference_map)
%% CRF refinement
if ~exist('.ppmimg') mkdir('.ppmimg'); end
if ~exist('.mat_dcl') mkdir('.mat_dcl'); end
if ~exist('.dcl_crf') mkdir('.dcl_crf'); end

%% write ppm image
imwrite(img, ['.ppmimg/', imgfilename, '.ppm']);   
%% save mat file for crf
data = single(zeros(config.CRF.im_sz,config.CRF.im_sz,2));
inference_map = inference_map';

data(1:size(inference_map,1), 1:size(inference_map,2),1) = inference_map;
data(1:size(inference_map,1), 1:size(inference_map,2),2) = inference_map;
save(['.mat_dcl/', imgfilename, '_blob_0.mat'], 'data');
CRF_CMD = sprintf('%s -id ./.ppmimg -fd ./.mat_dcl -sd ./.dcl_crf -i %d -px %d -py %d -pw %d -bx %d -by %d -br %d -bg %d -bb %d -bw %d',...
    config.CRF.CRF_DIR, config.CRF.CRF_ITER, config.CRF.px, config.CRF.py, config.CRF.pw, config.CRF.bx, config.CRF.by, config.CRF.br, config.CRF.bg, config.CRF.bb, config.CRF.bw);
system(CRF_CMD);
dcl_crf = -(LoadBinFile(['.dcl_crf/', imgfilename, '.bin'], 'float'));
dcl_crf = (dcl_crf - min(dcl_crf(:))) /(max(dcl_crf(:)) - min(dcl_crf(:)) + eps);
dcl_crf = 1.0./(1+exp(-5.0*(double(dcl_crf)-0.5)));
dcl_crf = uint8(dcl_crf.*255);

%% delete temporary files
delete('.ppmimg/*.*');
delete('.mat_dcl/*.*');
delete('.dcl_crf/*.*');

if ~exist('.ppmimg') rmdir('.ppmimg'); end
if ~exist('.mat_dcl') rmdir('.mat_dcl'); end
if ~exist('.dcl_crf') rmdir('.dcl_crf'); end
end
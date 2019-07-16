root_inference = '/home/ty/code/tf_saliency_attention/total_result/result_rnn_2019-07-16 17:31:35/';
root = '/home/ty/data/FBMS/FBMS_Testset/';
% root = '/home/ty/data/davis/480p/';
parent_file = dir(root_inference);
res_path = '/home/ty/code/tf_saliency_attention/total_result/result_rnn_crf_2019-07-16 17:31:35/';
mkdir(res_path);
fold_start = 3;

im_suffix = '.jpg';
map_suffix = '.png';
prior_suffix = '.png';

config = config_dcl_saliency();

for foldId = fold_start:length(parent_file)
    im_path = parent_file(foldId).name;
    files = dir([root im_path '/*' im_suffix]);
    files_inference = dir([root_inference im_path '/*' map_suffix]);
    test_start = 1;
    test_end = length(files_inference);
    test_num = test_end - test_start + 1;
    if test_num < 1
        continue;
    end
    
    mkdir([res_path im_path]);  
    for imInd = test_start:test_end
        name = files_inference(imInd).name(1:end-length(im_suffix));
        img = imread([root im_path '/' name im_suffix]);
        max_scale = max(size(img, 1), size(img, 2));
        if max_scale > 510
            img = imresize(img, 510/max_scale);
        end

        tic;
        inference_map = imread([root_inference im_path '/' name prior_suffix]);
        inference_map = double(inference_map) / 255;
        inference_map2 = (inference_map - min(inference_map(:))) /(max(inference_map(:)) - min(inference_map(:)) + eps);
        %[~,imgfilename,~] = fileparts(im_names{j});
        crf_smap =  do_crf_refinement(config, name, img, inference_map2);
        %imshow(crf_smap, [res_path im_path '/' name map_suffix]);
        crf_smap(crf_smap < 30) = 0;
        crf_smap(crf_smap > 220) = 255;
        imwrite(crf_smap, [res_path im_path '/' name map_suffix]);
        fprintf('Time cost: %.2f s\n', toc);
    end
end

exit()
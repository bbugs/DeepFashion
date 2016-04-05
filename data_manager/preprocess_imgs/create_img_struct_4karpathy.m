
%% Load csv with image info
clear all;
target_split = 'train';
fname = sprintf('../../data/fashion53k/csv/imgs/imgs_info_%s.csv', target_split);

% img_id	split	folder	img_filename
fid = fopen(fname, 'rt');
C = textscan(fid, '%s%s%s%s', 'HeaderLines', 1, 'CollectOutput', true, 'delimiter', ',');
fclose(fid);

img_ids = C{1,1}(:,1);
splits = C{1,1}(:,2);
folders = C{1,1}(:,3);
img_filenames = C{1,1}(:,4);

%%
% Load cnn file
n_regions = 4;

fname = sprintf('../../data/fashion53k/img_regions/4_regions_cnn/per_split/cnn_fc7_%s.txt', target_split);

cnn = csvread(fname);

size(cnn);

%% Create structure

Img = cell(length(img_ids), 1);

i = 1; % index over images
j = 1;
k = 1;
bias = ones(n_regions+1,1);
for i = 1:length(img_ids)
    
    s = struct('img_id',[], 'splits', [], 'codes', [], 'folder', [], 'fname', []);
    
    cnn_slice = cnn(j:j+n_regions, :);
    
    s(1).img_id = img_ids{i};
    s(1).splits = splits{i};
    s(1).codes = [cnn_slice bias];
    s(1).folder = folders{i};
    s(1).fname = img_filenames{i};
    
    j = j + n_regions + 1;
    
    Img{k} = s;  
    k = k + 1;
    
end

%%
fname = sprintf('../../data/fashion53k/matlab_structs/split_%s_img_new.mat', target_split);
save(fname, 'Img', '-v7.3')



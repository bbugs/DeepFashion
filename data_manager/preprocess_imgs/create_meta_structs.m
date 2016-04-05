%% Load Sent
clear all;
target_split = 'train';
fname = sprintf('../../data/fashion53k/matlab_structs/no_zappos/split_%s_sent.mat', target_split);

load(fname, 'Sent')

%% Initialize sentence to image

stoi = zeros(length(Sent),1);
% stoi <4000x1>
% 1
% 1
% 1
% 1
% 1
% 2
% 2
% etc


%% Iterate over Sent

img_id_in_split = 1;

current_img_id = str2double(Sent{1}.img_id);

for i = 1:length(Sent)
    
    img_id = str2double(Sent{i}.img_id);
    
    if img_id ~= current_img_id
        img_id_in_split = img_id_in_split + 1;
        current_img_id = img_id;
    end
        
    stoi(i) = img_id_in_split;

end

%% Image to Sentence
n_imgs = 1000;

if strcmp(target_split, 'test')
    n_imgs = 1000;
elseif strcmp(target_split, 'val')
    n_imgs = 4000;
elseif strcmp(target_split, 'train')
    n_imgs = 53689 - 4000 - 1000;
end


itos = cell(n_imgs,1);
current_img_id = str2double(Sent{1}.img_id);

sent_ids = zeros();

k = 1;
img_index = 1;
for i = 1:length(Sent)
    img_id = str2double(Sent{i}.img_id);
    if img_id ~= current_img_id
        itos{img_index} = sent_ids;
        sent_ids = zeros();
        current_img_id = img_id;
        k = 1;
        img_index = img_index + 1;
    end
    sent_ids(k,1) = i;
    k = k + 1;
end
itos{img_index} = sent_ids; % make sure last item gets written
    
%%
meta = struct();
meta.split = target_split;
meta.dataset = 'fashion53k';

%% Save
% fout_name = 'split_test_meta..mat';

fout_name = sprintf('../../data/fashion53k/matlab_structs/meta/split_%s_meta.mat', target_split);

save(fout_name, 'itos', 'stoi', 'meta')

% itos  <800x1 cell>
% each cell 
% img_id [1;2;3;4;5]



% meta is a struct
% meta.split = 'train'
% meta.dataset= 'paskal1k'
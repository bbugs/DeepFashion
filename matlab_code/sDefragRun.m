%  Explore DeFragRun

%% from DeFragDriver
clear all;
%rng('shuffle');

DriverSetParams;
params

%%

% parameters can override certian setting
fappend = getparam(params, 'fappend', '');
lrreduce = getparam(params, 'lrreduce', 1.0);
milstart = getparam(params, 'milstart', params.maxepochs * 0.75);

setupActFunc; % sets params.f and params.df using params.actFunc

%%
fprintf('loading word vectors...\n');
[oWe, ~, ~] = loadWordVectors(params);

fprintf('initializing params...\n');
initParams; % initialize parameters
[theta, decodeInfo] = param2stack(Wi2s, Wsem);
disp(['Size of full parameter vector: ' num2str(length(theta))])  
% sus: Size of full parameter vector: 3148600

%%

[~, node_name] = system('uname -n'); % get node name
node_name = node_name(1:end-1); % take out newline at end 
% sus: node_name 'Susana.local'

%%
% grab raw dataset data
fprintf('getting training fold...\n');
train_split = getDatasetFold(params, 'train');
fprintf('getting validation fold...\n');
val_split = getDatasetFold(params, 'val');
Ni = length(train_split.Img);
Ns = length(train_split.Sent);





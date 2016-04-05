
% file used in cross-validations

% create params struct from scratch
params = struct();

%% data
params.dataset = 'pascal1k';
params.maxepochs = 20;

%% misc practical params
params.fappend = '';
% minimum validation performance in order to save a model checkpoint
% (to conservedisk space... )
if strcmp(params.dataset, 'pascal1k'), params.min_save_score = 80; end
if strcmp(params.dataset, 'flickr8k'), params.min_save_score = 36; end
if strcmp(params.dataset, 'flickr30k'), params.min_save_score = 36; end

%% image fragments params
params.imgDim = 4097; % 4096 + 1 for bias...
params.viswordsmax = 20;
params.viswordsmin = 20; % take top 20 always for every image
params.viswordsthr = - 0.5;

%% sentence fragments params
params.word_dim = 200;
params.smoothnum = randi(10, 1); % smoothing, so that sentences with single word dont get huge scores because normalizer is 1
params.l2norm = 0;

%% optimization params
params.momentum = 0.9; % for sgd
params.lrreduce = 0.9; % as a fraction of maxiters, LR will be decreased x 0.1
params.lr = 10.^(-6.5 + (rand()*3-1.5));
params.regC = 10.^(rand()*4 - 10);
params.batch_size = randi(40)+10;
params.method = 0; % sgd

%% defrag model setup
tt = 2;
if tt==0, params.uselocal = true; params.useglobal = false; end
if tt==1, params.uselocal = false; params.useglobal = true; end
if tt>=2, params.uselocal = true; params.useglobal = true; end
params.usemil = true;
params.actFunc = 'rectMax';

params.h = randi(700)+500; % size of semantic space

params.gmargin = 10.^(rand()*2);
params.gscale = 10.^(rand()*2-2);

params.lmargin = 1;
params.lscale = 1;

params.thrglobalscore = 1; % when computing global score, threshold fragment scores at 0?
params.maxaccum = false;

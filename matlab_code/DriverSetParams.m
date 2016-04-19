 % create params struct from scratch
params = struct();

%% data
params.dataset = 'pascal1k';

%% misc practical params
params.fappend = '';
% minimum validation performance in order to save a model checkpoint
% (to conservedisk space... ). score is computed as average of R@10
% performance
if strcmp(params.dataset, '../data/pascal1k'), params.min_save_score = 80; end
if strcmp(params.dataset, 'flickr8k'), params.min_save_score = 40; end
if strcmp(params.dataset, 'flickr30k'), params.min_save_score = 40; end

%% image fragments params
params.imgDim = 4097; % 4096 + 1 for bias...
params.viswordsmax = 20;
params.viswordsmin = 20; % take top 20 always for every image
params.viswordsthr = - 0.5;

%% sentence fragments params
params.word_dim = 200; % dim of word vectors
params.smoothnum = 10; % smoothing, so that sentences with single word dont get huge scores because normalizer is 1
params.l2norm = 0; % force l2 normalize word vectors?

%% optimization params
params.maxepochs = 20; % numer of epochs through data
params.momentum = 0.9; % for sgd
params.lrreduce = 1; % as a fraction of maxiters, LR will be decreased x 0.1. 1 = dont
params.lr = 1e-5; % learning rate
params.regC = 2e-10; % L2 regularization strength
params.batch_size = 100; % for minibatches
params.method = 0; % sgd. 1 = adadelta, 2 = adagrad

%% defrag model setup
tt = 2;
if tt==0, params.uselocal = true; params.useglobal = false; end
if tt==1, params.uselocal = false; params.useglobal = true; end
if tt>=2, params.uselocal = true; params.useglobal = true; end
params.usemil = true;
params.actFunc = 'rectMax';

params.h = 1000; % size of semantic space

params.gmargin = 40;
params.gscale = 0.5;

params.lmargin = 1;
params.lscale = 1;

params.thrglobalscore = 0; % when computing global score, threshold fragment scores at 0?
params.maxaccum = true;

%% susana params
params.sus_depTree = false;
params.sus_toy = false;
params.dataset = 'fashion53k';
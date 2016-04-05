clear all;

%% Explore DeFragTestEval
checkpoint = 'cv/example_pascal1k_checkpoint.mat';


load(checkpoint, 'report');
params = report.params;  %sus report is the name of the checkpoint
setupActFunc;
[oWe, ~, ~] = loadWordVectors(params);  %sus: precomputed word vectors either glove, mikolov or BiRNN.
%sus: oWe: 200x400000. The number or hidden dimensions is 200.

test_split = getDatasetFold(params, 'test');

% sus:
% test_split is a 1x1 struct, containing the following:
% test_split.meta.split  % 'test'
% test_split.meta.dataset % 'pascal1k'
% tes_split.itos. This image to sentence is a 100x1 cell. 100 test images. For each test image, specify the
% sentences that correspond to it. 
% stoi: sentence to image. 500x1 Each row is sentence (500 sentences 
% corresponding to 1000 test images)

% test_split.Img{1,1}.codes  20x4097  I suppose These are the weights, from Imagenet
%  



%% Explore DeFragEval
% [e2r, e3r] = DeFragEval(test_split, oWe, params, report.theta, report.decodeInfo);

finetuneCNN = false;
thrglobalscore = params.thrglobalscore;  %sus: 1
smoothnum = params.smoothnum;  %sus: 10
maxaccum = params.maxaccum;  %sus: 0 in default params
theta = report.theta;  %sus theta <3,148,600 x 1>. This corresponds to 700*(4097+401).
% 401 is because two word embeddings are concatenated, each of 200d plus a
% bias term
decodeInfo = report.decodeInfo;  %sus decodeInfo = [700,4097],[700,401]
%sus h = 700: size of semantic space

% decode parameters
[Wi2s, Wsem] = stack2param(theta, decodeInfo);  %convert theta into two matrices of sizes [700,4097],[700,401]

Ni = length(test_split.Img);  % 100 images
Ns = length(test_split.Sent); % 500 sentences

% forward images, ie, project each image fragment (20) into the 700-dim
% space
allImgVecsCell = cell(1, Ni);  
imgVecICell = cell(1, Ni);  
for i = 1:Ni
    codes = test_split.Img{i}.codes;  % extract CNN representation
    allImgVecsCell{i} = Wi2s * codes';
    imgVecICell{i} = ones(size(codes,1),1) * i; %sus: some id keeping
end
% after the forward pass, %<1x100 cell>. Each cell is size <700x20>. 
% Each image fragment (20) is represented as 700d vector

%%
% Forward sentence fragments and compute global image-sentence alignment
M = zeros(Ns, Ni);
for i=1:Ns  %for each sentence
    %z is the sentence representation as in eq. 1 in Nips paper. We have to
    %change the ForwardSent and BackwardSents to change to the bidrectional
    %RNN as in CVPR paper
    [z, ~] = ForwardSent(test_split.Sent{i}, params, oWe, Wsem); % z is 700 x number of words in sentence
    if isempty(z) % this can happen if sentence is very short etc.
        M(i,:) = -inf;
        continue
    end
    for j=1:Ni  % for each image
        d = allImgVecsCell{j}' * z;  %sus: dot product between image vectors (20x700) and word vectors (700x number_of_words_in_sentence)
        
        if thrglobalscore, d(d<0) = 0; end  %treshold d at zero (no negative values)
        
        if maxaccum  %set this to 1 for his CVPR implementation 
            s = sum(max(d, [], 1)); %scores
        else
            s = sum(d(:));
        end
        
        s = s / (size(d,2) + smoothnum);
        M(i,j) = s; %M is 500x100. It represents the scores of each sentence with each image
    end
end


imageIDs = test_split.stoi;

% fix sentence, rank images
E2ranks = zeros(size(M,1),1);
for s=1:size(M,1)
    [d,ind] = sort(M(s,:), 'descend');
    thisImageID = imageIDs(s);
    rankOfClosest = find(thisImageID==ind,1);
    E2ranks(s) = rankOfClosest;  %500 x 1: for each sentence, get the highest ranked image
end

M = M'; % flip around, rank other way
E3ranks = zeros(size(M,1),1);
for i=1:size(M,1)
    [d,ind] = sort(M(i,:), 'descend');
    thisImageID = i;
    imageIDsOfRankedSentences = imageIDs(ind);
    rankOfClosest = find(thisImageID==imageIDsOfRankedSentences,1);
    E3ranks(i) = rankOfClosest;  %100 x 1: for each image, get the highest ranked sentence
end



% fprintf('%s test performance from %s:\n', params.dataset, checkpoint);
% fprintf('e2 (image search)    : r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e2r<=1)*100, mean(e2r<=5)*100, mean(e2r<=10)*100, mean(e2r), floor(median(e2r)));
% fprintf('e3 (image annotation): r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e3r<=1)*100, mean(e3r<=5)*100, mean(e3r<=10)*100, mean(e3r), floor(median(e3r)));
% fprintf('-----\n');

% sus:
% e2 (image search)    : r@1: 29.4,	 r@5: 67.8,	 r@10: 82.6,	 rAVG: 7.4,	 rMED: 3
% e3 (image annotation): r@1: 35.0,	 r@5: 67.0,	 r@10: 86.0,	 rAVG: 10.6,	 rMED: 3

function [E2ranks, E3ranks] = DeFragEval(split, oWe, params, theta, decodeInfo, rcnn_model)

finetuneCNN = false;
thrglobalscore = params.thrglobalscore;
smoothnum = params.smoothnum;
maxaccum = params.maxaccum;

% decode parameters
[Wi2s, Wsem] = stack2param(theta, decodeInfo);

Ni = length(split.Img);
Ns = length(split.Sent);

% Forward image fragments
if finetuneCNN
    % forward images in batches of up to 12 through the CNN. 12 because
    % 12*20 < 256. and 20 is number of dets per image
    allImgVecsCell = cell(1, Ni);
    imgVecICell = cell(1, Ni);
    for i = 1:12:Ni
        for j=0:11
            if i+j>Ni, break; end
            assert(size(split.Imgs{i+j}.dets, 1) == 20);
        end
        imax = i+12-1;
        if imax>Ni, imax=Ni; end
        codes = andrej_forwardRCNN(split.Imgs(i:imax), rcnn_model, params);
        for j=0:(imax-i)
            allImgVecsCell{i+j} = Wi2s * codes(j*20+1:(j+1)*20, :)';
            imgVecICell{i+j} = ones(20,1)*(i+j);
        end
    end
else
    % forward images
    allImgVecsCell = cell(1, Ni);
    imgVecICell = cell(1, Ni);
    for i = 1:Ni
        codes = split.Img{i}.codes;
        allImgVecsCell{i} = Wi2s * codes';
        imgVecICell{i} = ones(size(codes,1),1) * i;
    end
end

% Forward sentence fragments and compute global image-sentence alignment
M = zeros(Ns, Ni);
for i=1:Ns
    [z, ~] = ForwardSent(split.Sent{i}, params, oWe, Wsem);
    if isempty(z) % this can happen if sentence is very short etc.
        M(i,:) = -inf;
        continue
    end
    for j=1:Ni
        d = allImgVecsCell{j}' * z;
        
        if thrglobalscore, d(d<0) = 0; end
        
        if maxaccum
            s = sum(max(d, [], 1));
        else
            s = sum(d(:));
        end
        
        s = s / (size(d,2) + smoothnum);
        M(i,j) = s;
    end
end

imageIDs = split.stoi;

% fix sentence, rank images
E2ranks = zeros(size(M,1),1);
for s=1:size(M,1)
    [d,ind] = sort(M(s,:), 'descend');
    thisImageID = imageIDs(s);
    rankOfClosest = find(thisImageID==ind,1);
    E2ranks(s) = rankOfClosest;
end

M = M'; % flip around, rank other way
E3ranks = zeros(size(M,1),1);
for i=1:size(M,1)
    [d,ind] = sort(M(i,:), 'descend');
    thisImageID = i;
    imageIDsOfRankedSentences = imageIDs(ind);
    rankOfClosest = find(thisImageID==imageIDsOfRankedSentences,1);
    E3ranks(i) = rankOfClosest;
end

end

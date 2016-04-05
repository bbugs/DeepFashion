function [cost_struct, grad, df_CNN] = DeFragCost(theta,decodeInfo,params, oWe,imgFeats,depTrees, rcnn_model)
% returns cost, gradient, and gradient wrt image vectors which
% can be forwarded to the CNN for finetuning outside of this

domil = getparam(params, 'domil', false);
useglobal = getparam(params, 'useglobal', true);
uselocal = getparam(params, 'uselocal', true);
gmargin = params.gmargin;
lmargin = params.lmargin;
gscale = params.gscale;
lscale = params.lscale;
thrglobalscore = params.thrglobalscore;
smoothnum = params.smoothnum;
maxaccum = params.maxaccum;

finetuneCNN = false;
df_CNN = 0;

% unpack parameters
[Wi2s, Wsem] = stack2param(theta, decodeInfo);
cost = 0;

N = length(depTrees); % number of samples

% forward prop all image fragments and arrange them into a single large matrix
imgVecsCell = cell(1, N);
imgVecICell = cell(1, N);
if finetuneCNN
    % forward RCNN
    imgVecs = andrej_forwardRCNN(imgFeats, rcnn_model, params);
    for i=1:N
        imgVecICell{i} = ones(size(imgFeats{i}.codes,1), 1)*i;
    end
else
    for i=1:N        
        imgVecsCell{i} = imgFeats{i}.codes;
    end
    imgVecs = cat(1, imgVecsCell{:});
    for i=1:N
        imgVecICell{i} = ones(size(imgVecsCell{i},1), 1)*i;
    end
end
imgVecI = cat(1, imgVecICell{:});
allImgVecs = Wi2s * imgVecs'; % the mapped vectors are now columns
Ni = size(allImgVecs,2);

% forward prop all sentences and arrange them
sentVecsCell = cell(1, N);
sentTriplesCell = cell(1, N);
sentVecICell = cell(1, N);
for i = 1:N
    [z, ts] = ForwardSent(depTrees{i},params,oWe,Wsem);
    sentVecsCell{i} = z;
    sentTriplesCell{i} = ts;
    sentVecICell{i} = ones(size(z,2), 1)*i;
end
sentVecI = cat(1, sentVecICell{:});
allSentVecs = cat(2, sentVecsCell{:});
Ns = size(allSentVecs, 2);

% compute fragment scores
dots = allImgVecs' * allSentVecs;

% compute local objective
if uselocal

    MEQ = bsxfun(@eq, imgVecI, sentVecI'); % indicator array for what should be high and low
    Y = -ones(size(MEQ));
    Y(MEQ) = 1;

    if domil

        % miSVM formulation: we are minimizing over Y in the objective,
        % what follows is a heuristic for it mentioned in miSVM paper.
        fpos = dots .* MEQ - 9999  * (~MEQ); % simplifies things
        Ypos = sign(fpos);

        ixbad = find(~any(Ypos==1,1));
        if ~isempty(ixbad)
            [~, fmaxi] = max(fpos(:,ixbad), [], 1);
            Ypos = Ypos + sparse(fmaxi, ixbad, 2, Ni, Ns); % flip from -1 to 1: add 2
        end

        Y(MEQ) = Ypos(MEQ); % augment Y in positive bags
    end

    % weighted fragment alignment objective
    marg = max(0, lmargin - Y .* dots); % compute margins
    W = zeros(Ni, Ns);
    for i=1:Ns
        ypos = Y(:,i)==1;
        yneg = Y(:,i)==-1;
        W(ypos, i) = 1/sum(ypos);
        W(yneg, i) = 1/(sum(yneg));
    end
    wmarg = W .* marg;
    lcost = lscale * sum(wmarg(:));
    cost = cost + lcost;
end

% compute global objective
if useglobal
    
    % forward scores in all regions
    SG = zeros(N,N);
    SGN = zeros(N,N); % the number of values (for mean)
    accumsis = cell(N,N);
    for i=1:N
        for j=1:N
            d = dots(imgVecI == i, sentVecI == j);
            if thrglobalscore, d(d<0) = 0; end
            if maxaccum
                [sv, si] = max(d, [], 1); % score will be max (i.e. we're finding support of each fragment in image)
                accumsis{i,j} = si; % remember switches for backprop
                s = sum(sv);
            else
                s = sum(d(:)); % score is sum
            end
            nnorm = size(d,2); % number of sent fragments
            nnorm = nnorm + smoothnum;
            s = s/nnorm;
            SG(i,j) = s;
            SGN(i,j) = nnorm;
        end
    end
    
    % compute the cost
    gcost = 0;
    cdiffs = zeros(N,N);
    rdiffs = zeros(N,N);
    for i=1:N
        % i is the pivot. It should have higher score than col and row
        
        % col term
        cdiff = max(0, SG(:,i) - SG(i,i) + gmargin);
        cdiff(i) = 0; % nvm score with self
        cdiffs(:, i) = cdiff; % useful in backprop
        
        % row term
        rdiff = max(0, SG(i,:) - SG(i,i) + gmargin);
        rdiff(i) = 0;
        rdiffs(i, :) = rdiff; % useful in backprop
        
        gcost = gcost + sum(cdiff) + sum(rdiff);
    end
    
    gcost = gscale * gcost;
    cost = cost + gcost;
end

ltop = zeros(Ni, Ns);

if uselocal
    % backprop local objective
    ltop = ltop - lscale * (marg > 0) .* Y .* W;
end

if useglobal
    % backprop global objective
    
    % backprop margin
    dsg = zeros(N,N);
    for i=1:N
        cd = cdiffs(:,i);
        rd = rdiffs(i,:);
        
        % col term backprop
        dsg(i,i) = dsg(i,i) - sum(cd > 0);
        dsg(:,i) = dsg(:,i) + (cd > 0);
        
        % row term backprop
        dsg(i,i) = dsg(i,i) - sum(rd > 0);
        dsg(i,:) = dsg(i,:) + (rd > 0);
    end
    
    % backprop into scores
    ltopg = zeros(size(ltop));
    for i=1:N
        for j=1:N
            
            % backprop through the accumulation function
            if maxaccum
                % route the gradient along in each column. bit messy...
                gradroute = dsg(i,j) / SGN(i,j);
                mji = find(sentVecI == j);
                mii = find(imgVecI == i);
                accumsi = accumsis{i,j};
                for q=1:length(mji)
                    miy = mii(accumsi(q));
                    mix = mji(q);
                    if thrglobalscore
                        if dots(miy,mix) > 0
                            ltopg(miy, mix) = gradroute;
                        end
                    else
                        ltopg(miy, mix) = gradroute;
                    end
                end
            else
                d = dots(imgVecI == i, sentVecI == j);
                dd = ones(size(d)) * dsg(i,j) / SGN(i,j);
                if thrglobalscore
                    dd(d<0) = 0;
                end
                ltopg(imgVecI == i, sentVecI == j) = dd;
            end
        end
    end
    ltop = ltop + gscale * ltopg;
end

% backprop into fragment vectors
allDeltasImg = allSentVecs * ltop';
allDeltasSent = allImgVecs * ltop;

% backprop image mapping
df_Wi2s = allDeltasImg * imgVecs;

if finetuneCNN
    % derivative wrt CNN data so that we can pass on gradient to RCNN
    df_CNN = allDeltasImg' * Wi2s;
end

% backprop sentence mapping
df_Wsem = BackwardSents(depTrees,params,oWe,Wsem,sentVecsCell,allDeltasSent);

cost_struct = struct();
cost_struct.raw_cost = cost;
cost_struct.reg_cost = params.regC/2 * sum(theta.^2);
cost_struct.cost = cost_struct.raw_cost + cost_struct.reg_cost;

%[grad,~] = param2stack(df_Wi2s, df_Wsem);
grad = [df_Wi2s(:); df_Wsem(:);]; % for speed hardcode param2stack
grad = grad + params.regC * theta; % regularization term gradient
end
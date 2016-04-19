function DeFragRun(params)
% This function is concerned with the optimization. It calls the cost function,
% performs parameter updates and writes intermediate models if performance is good.
% theta and decodeInfo can be passed in as initial solution

% parameters can override certian setting
fappend = getparam(params, 'fappend', '');
lrreduce = getparam(params, 'lrreduce', 1.0);
milstart = getparam(params, 'milstart', params.maxepochs * 0.75);

setupActFunc; % sets params.f and params.df using params.actFunc

fprintf('loading word vectors...\n');
[oWe, ~, ~] = loadWordVectors(params);  %sus: no need to change this function. vectors are loaded as they are oWe: 200x400000

fprintf('initializing params...\n');
initParams; % initialize parameters Wsem and Wi2s
[theta, decodeInfo] = param2stack(Wi2s, Wsem); %sus theta <3,148,600 x 1>. This corresponds to 700*(4097+401).
disp(['Size of full parameter vector: ' num2str(length(theta))])

[~, node_name] = system('uname -n'); % get node name
node_name = node_name(1:end-1); % take out newline at end    

% grab raw dataset data
fprintf('getting training fold...\n');
train_split = getDatasetFold(params, 'train');
fprintf('getting validation fold...\n');
val_split = getDatasetFold(params, 'val');
Ni = length(train_split.Img);  % sus 100 images
Ns = length(train_split.Sent);  % sus 500 sentences

% for validation we will keep track of both train/val error. We want to use
% the same number of items so that the scores are comparable. Thus,
% downsample the training split to create a small training split to eval
train_split_eval = splitSubsample(train_split, length(val_split.Img));

maxiters = ceil((Ns/params.batch_size)*params.maxepochs);
ppi = 1; % how often (in units of epochs) to evaluate validation error and (maybe) save checkpoints
pp = ceil(ppi*(Ns/params.batch_size)); % convert ppi from units in epochs to units in interations

iter = 1;
Eg = zeros(size(theta)); % accumulators for momentum/adadelta etc
Ex = zeros(size(theta));
raw_costs = zeros(maxiters, 1);
reg_costs = zeros(maxiters, 1);

best_score = getparam(params, 'min_save_score', -1);

hist_iter = [];
hist_e2v = []; % validation scores
hist_e3v = [];
hist_e2t = []; % training scores
hist_e3t = [];

fprintf('starting optimization...\n');
while iter <= maxiters
    
    % modulate MIL in params
    current_epoch = iter/maxiters*params.maxepochs;
    if params.usemil
        if current_epoch > milstart
            params.domil = true;
        else
            params.domil = false;
        end
    else
        params.domil = false;
    end
    
    % sample a image-sentence batch
    rp = randperm(Ni);
    rpb = rp(1:params.batch_size);
    rpbs = zeros(params.batch_size, 1);
    for q=1:params.batch_size
        six = train_split.itos{rpb(q)}; % sentence indeces for this image
        rpbs(q) = six(randi(length(six), 1)); % take a random sentence
    end
    allImgVecs_batch = train_split.Img(rpb);
    allTreesTrain_batch = train_split.Sent(rpbs);
    
    % evaluate cost and gradient! All magic happens inside
    [cost, grad] = DeFragCost(theta,decodeInfo,params,oWe,allImgVecs_batch,allTreesTrain_batch);
    
    % learning rate modulation
    lrmod = 1;
    if iter/maxiters > lrreduce
        lrmod = 0.1;
    end
    
    if params.method == 0
        
        % sgd momentum
        dx = params.momentum * Eg - lrmod * params.lr * grad;
        Eg = dx;
        
    elseif params.method == 1
        
        % adadelta update rules
        Eg = params.ro*Eg + (1-params.ro)*(grad.^2);
        dx = - sqrt((Ex+params.epsilon)./(Eg+params.epsilon)) .* grad;
        Ex = params.ro*Ex + (1-params.ro)*(dx.^2);

    elseif params.method == 2
    
        % adagrad update rules
        Eg = Eg + grad.^2;
        dx = - params.lr * grad ./ (sqrt(Eg+params.epsilon));
    end
    
    % perform parameter update
    theta = theta + dx;
    
    % keep track of costs
    raw_cost = cost.raw_cost;
    reg_cost = cost.reg_cost;
    total_cost = cost.cost;
    fprintf('iter %d/%d (%.1f%% done): raw_cost: %f \t reg_cost: %f \t cost: %f\n', iter, maxiters, 100*iter/maxiters, raw_cost, reg_cost, total_cost);
    raw_costs(iter) = raw_cost;
    reg_costs(iter) = reg_cost;
    
    if isnan(raw_cost + reg_cost),
        fprintf('WARNING! COST WAS NAN? ABORTING.\n');
        break; % something blew up, get out
    end
    
    if iter==1, score0 = total_cost; end % remember cost in beginning
    if total_cost > score0 * 10
        fprintf('WARNING! COST SEEMS TO BE EXPLODING. ABORTING.\n');
        break; % we're exploding: learning rate too high or something. get out
    end
    
    % evaluate validation performance every now and then, or on final iteration
    if (mod(iter, pp) == 0 && iter < 0.99*maxiters) || iter == maxiters
        
        % we will construct a REPORT strucutre that contains much of what
        % has happened during the training.
        report = struct;
        report.raw_costs = raw_costs;
        report.reg_costs = reg_costs;
        report.iter = iter;
        report.maxiters = maxiters;
        
        % evaluate and record validation set performance
        [e2r, e3r] = DeFragEval(val_split, oWe, params, theta, decodeInfo);
        fprintf('validation performance:\n');
        fprintf('e2: r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e2r<=1)*100, mean(e2r<=5)*100, mean(e2r<=10)*100, mean(e2r), floor(median(e2r)));
        fprintf('e3: r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e3r<=1)*100, mean(e3r<=5)*100, mean(e3r<=10)*100, mean(e3r), floor(median(e3r)));
        fprintf('-----\n');
        score = (mean(e2r<=10)*100 + mean(e3r<=10)*100)/2; % take average R@10 as score
        hist_iter(end+1) = iter;
        hist_e2v(end+1) = mean(e2r<=10)*100;
        hist_e3v(end+1) = mean(e3r<=10)*100;
        
        report.val_e2r = e2r;
        report.val_e3r = e3r;
        report.hist_iter = hist_iter;
        report.hist_e2v = hist_e2v;
        report.hist_e3v = hist_e3v;
        report.val_score = score;
        
        % evaluate training set based on random subset of examples equal in
        % size to the validation set
        [e2r, e3r] = DeFragEval(train_split_eval, oWe, params, theta, decodeInfo);
        fprintf('training performance:\n');
        fprintf('e2: r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e2r<=1)*100, mean(e2r<=5)*100, mean(e2r<=10)*100, mean(e2r), floor(median(e2r)));
        fprintf('e3: r@1: %.1f,\t r@5: %.1f,\t r@10: %.1f,\t rAVG: %.1f,\t rMED: %d\n', mean(e3r<=1)*100, mean(e3r<=5)*100, mean(e3r<=10)*100, mean(e3r), floor(median(e3r)));
        fprintf('-----\n');
        score = (mean(e2r<=10)*100 + mean(e3r<=10)*100)/2; % take average R@10 as score
        hist_e2t(end+1) = mean(e2r<=10)*100;
        hist_e3t(end+1) = mean(e3r<=10)*100;
        
        report.train_e2r = e2r;
        report.train_e3r = e3r;
        report.hist_e2t = hist_e2t;
        report.hist_e3t = hist_e3t;
        report.train_score = score;
        
        % report results to results file
        writeTextFile('cv/defrag_results.txt', sprintf('[%s] %s: iteration %d/%d (%.2f%% done) val score %f, train score %f', datestr(clock, 0), node_name, iter, maxiters, 100*iter/maxiters, report.val_score, report.train_score));
        
        p = params; % make a params copy
        p.f = 0; % take out the functions, they cause trouble when we try to save a function into .mat file
        p.df = 0;
        report.params = p;
        
        if iter == maxiters
            % this is the last iteration. Lets save a record of how it went
            top_val_score = max(0.5*(hist_e2v + hist_e3v));
            top_train_score = max(0.5*(hist_e2t + hist_e3t));
            randnum = floor(rand()*10000);
            if ~isempty(fappend)
                fsave = sprintf('cv/endreport_%s_%s_%d_%.0f_%.0f.mat', fappend, params.dataset, randnum, top_val_score, top_train_score);
            else
                fsave = sprintf('cv/endreport_%s_%d_%.0f_%.0f.mat', params.dataset, randnum, top_val_score, top_train_score);
            end
            save(fsave, 'report'); % save the report, but no need for the actual parameter vector theta
            writeTextFile('cv/defrag_results.txt', sprintf('%s: saved end report to %s', node_name, fsave));
        end
        
        % check if the performance is best so far, and if so save results
        if report.val_score > best_score
            best_score = report.val_score;
            
            % back up parameters into checkpoint
            report.theta = theta;
            report.decodeInfo = decodeInfo;
            
            if ~isempty(fappend)
                fsave = sprintf('cv/defrag_%s_%s_%s_m%d_l%d_g%d_%.0f.mat', fappend, params.dataset, node_name, params.usemil, params.uselocal, params.useglobal, report.val_score);
            else
                fsave = sprintf('cv/defrag_%s_%s_m%d_l%d_g%d_%.0f.mat', params.dataset, node_name, params.usemil, params.uselocal, params.useglobal, report.val_score);
            end
            writeTextFile('cv/defrag_results.txt', sprintf('%s: saved checkpoint to %s', node_name, fsave));
            
            % save checkpoint
            save(fsave, 'report');
        end
    end
    
    % phew, lets go on now
    iter = iter + 1;
end


end
function split = getDatasetFold(params, fold)
% fold is 'val', 'train', or 'test'

dataset = params.dataset;

% Assemble the split: we store everything independently for more modularity
load(sprintf('%s/split_%s_meta.mat', dataset, fold), 'meta', 'itos', 'stoi');
load(sprintf('%s/split_%s_sent_deptrees.mat', dataset, fold), 'Sent');
load(sprintf('%s/split_%s_img.mat', dataset, fold), 'Img');

% assemble the split
split = struct;
split.meta = meta;
split.itos = itos;
split.stoi = stoi;
split.Sent = Sent;
split.Img = Img;

% respect user setting in filtering detections
% this allows us to take detections above some score, but also 
% constrain the returned number to some range of [min, max]
if isfield(params, 'viswordsmin')
    if params.viswordsmin < 20
        fprintf('filtering image detections by minmax [%d, %d] and threshold %f\n' ...
               , params.viswordsmin, params.viswordsmax, params.viswordsthr);

        Ni = length(split.Img);
        for q=1:Ni
            keep = true(size(split.Img{q}.codes,1), 1);
            if isfield(params, 'viswordsthr')
                keep = split.Img{q}.dets(:,5) > params.viswordsthr;
            end
            if isfield(params, 'viswordsmax')
                keep(params.viswordsmax+1:end) = false;
            end
            if isfield(params, 'viswordsmin')
                keep(1:params.viswordsmin) = true;
            end
            if ~all(keep)
                % filter
                split.Img{q}.codes = split.Img{q}.codes(keep, :);
                split.Img{q}.dets = split.Img{q}.dets(keep, :);
            end
        end
    end
end

end

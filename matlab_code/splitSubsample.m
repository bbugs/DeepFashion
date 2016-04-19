function splitsub = splitSubsample(split, num, maxsent)
% takes data in a split and creates a smaller split out of the data,
% with num examples only

splitsub = struct();

% take first num items
splitsub.Img = split.Img(1:num);

SentCell = cell(num, 1);
stoiCell = cell(num, 1);
for i=1:num
    
    six = split.itos{i}; % corresponding sentences
    if nargin > 2
        % only take first maxsent
        six = six(1:maxsent);
    end
    SentCell{i} = split.Sent(six); % fetch the appropriate sentences
    stoiCell{i} = ones(length(six), 1) * i;
end
splitsub.Sent = cat(2, SentCell{:});
splitsub.stoi = cat(1, stoiCell{:});
splitsub.itos = helperInvertMap(splitsub.stoi);

meta = split.meta;
meta.split = strcat(meta.split, '_subsampled');
splitsub.meta = meta;

end

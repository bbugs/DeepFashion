function [oWe, vocab, wordMap] = loadWordVectors(params)
    
    % load word vector mappings
    word_dim = 200;
    fsave = sprintf('common/wordvecs_%dd.mat', word_dim);
    
    if nargout == 1 % for speed
        vocab = 0;
        wordMap = 0;
        load(fsave, 'oWe');
    else
        load(fsave, 'oWe', 'vocab');
        wordMap = containers.Map(vocab,1:length(vocab));
    end
    
    % l2 normalize word vectors if user wants to
    if params.l2norm
        fprintf('l2 normalizing word vectors to unit length...\n');
        oWe = bsxfun(@rdivide, oWe, sqrt(sum(oWe.^2, 1)));
    end
    
end

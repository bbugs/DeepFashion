% sus: Bring to workspace Wsem <700x401> and Wi2s <700x4097>

% if statement added by susana (remove 2 since we deal with BiRNN and 
% not dependency trees):
if params.sus_depTree
    % if original setup with dependency Trees
    fanIn = 2 * params.word_dim + 1; % +1 for bias
else
    % if we use only one word
    fanIn = params.word_dim + 1; % +1 for bias
end

fanOut = params.h;
range = 1/sqrt(6*fanIn + fanOut);
Wsem = -range + (2*range).*rand(params.h,fanIn);

% image side mapping is just a linear transform
fanIn = params.imgDim;  % 4097
fanOut = params.h ;  %700
range = 1/sqrt(6*fanIn + fanOut);
Wi2s = -range + (2*range).*rand(fanOut,fanIn);


function [stack decodeInfo decodeIx] = param2stack(varargin)

stack = [];

n = 1;
for i=1:length(varargin)
    if iscell(varargin{i})
        ixCell = {};
        for c = 1:length(varargin{i})
            decodeCell{c} = size(varargin{i}{c});
            nelt = numel(varargin{i}{c});
            ixCell{c} = n:n+nelt-1;
            n = n + nelt;
            stack = [stack ; varargin{i}{c}(:)];
        end
        decodeIx{i} = ixCell;
        decodeInfo{i} = decodeCell;
        clear decodeCell;
    else
        decodeInfo{i} = size(varargin{i});
        nelt = numel(varargin{i});
        decodeIx{i} = n:n+nelt-1;
        n = n + nelt;
        stack = [stack ; varargin{i}(:)];
    end
end
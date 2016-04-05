function writeTextFile(fileName,cellOfStrings,opt) 
% writeTextFile(fileName,cellOfStrings,options)
% opt.writeFlag = 
%       'a'     open or create file for writing; append data to end of file
%       'r+'    open (do not create) file for reading and writing
%       'w+'    open or create file for reading and writing; discard
%               existing contents
%       'a+'    (default) open or create file for reading and writing; append data
%               to end of file
%       'W'     open file for writing without automatic flushing
%       'A'     open file for appending without automatic flushing
% opt.separator = 
%       '\n' (default) or ';' or ' ' etc.
% struct('separator','\t')
% struct('writeFlag','w+')
% struct('separator','\t','writeFlag','w+')
% richard _at_ socher .org


if ~exist('opt','var') || (exist('opt','var') && ~isfield(opt,'writeFlag'))
    opt.writeFlag='a+';
end

if ~exist('opt','var') || (exist('opt','var') && ~isfield(opt,'separator'))
    opt.separator='\n';
end
if isnumeric(cellOfStrings)
    % if input is a matrix, each row is written on one line
    fid = fopen(fileName, opt.writeFlag);
    for s = 1:size(cellOfStrings,1)
        fprintf(fid, ['%f '], cellOfStrings(s,:));
        fprintf(fid, ['\n']);
    end
    fclose(fid);
elseif ischar(cellOfStrings)
    % if input is a matrix, each row is written on one line
    fid = fopen(fileName, opt.writeFlag);
    for s = 1:size(cellOfStrings,1)
        fprintf(fid,'%s', cellOfStrings);
        fprintf(fid, ['\n']);
    end
    fclose(fid);
    
elseif iscell(cellOfStrings{1})
    fid = fopen(fileName, opt.writeFlag);
    for s = 1:length(cellOfStrings)
        for w = 1:length(cellOfStrings{s})
            fprintf(fid, ['%s' ' '], cellOfStrings{s}{w});
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
else
    
    fid = fopen(fileName, opt.writeFlag);
    for s = 1:length(cellOfStrings)
        if ischar(cellOfStrings{s})
            fprintf(fid, ['%s' opt.separator], cellOfStrings{s});
        else
            fprintf(fid, ['%s' opt.separator], num2str(cellOfStrings{s}));
        end
    end
    if ~strcmp(opt.separator,'\n')
        fprintf(fid, '\n');
    end
    fclose(fid);
end

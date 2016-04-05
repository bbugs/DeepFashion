function fileLines = readTextFile(fileName)
% richard _at_ socher .org

fid = fopen(fileName, 'r');
fileLines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 99900000);
fclose(fid);
fileLines = fileLines{1};

% % % fileLines = readTextFile('../data/training_set_rel3.tsv')
% % % %followed for instance by:
% for li = 1:length(fileLines)
% % startIndex, endIndex, tokIndex, matchStr, tokenStr, exprNames, splitStr] 
%     [~, ~, ~, ~, ~, ~, splitLine] = regexp(fileLines{li}, '\t');
% %...
% end
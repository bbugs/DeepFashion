%%
clear all;

%%
load ('struct_info.tmp.mat')  % loads a variable called struct_info

fieldnames = struct_info.fieldnames;

%% Load csv file with words per image
csv_table = readtable(struct_info.csv_fname);
% the csv file contains the fieldnames in fieldnames, e.g.,
% img_id,split,folder,img_filename,asin,word
%% Load fashion vocabulary
fname = struct_info.fashion_vocab_fname;
fid = fopen(fname, 'rt');
C = textscan(fid, '%s', 'HeaderLines', 0, 'CollectOutput', true);
fclose(fid);
dress_vocab = C{1};

%% Load glove or word2vec vocabulary
fname = struct_info.word2vec_vocab_fname;
fid = fopen(fname, 'rt');
C = textscan(fid, '%s', 'HeaderLines', 0, 'CollectOutput', true);
fclose(fid);
word2vec_vocab = C{1};

%% Create word2id_dress and id2word_dress and word2id_glove and id2word_glove
word2id_dress = containers.Map(dress_vocab,1:length(dress_vocab));
% word2id_dress('satin')
id2word_dress = containers.Map(1:length(dress_vocab), dress_vocab);

%%
word2id_glove = containers.Map(word2vec_vocab, 1:length(word2vec_vocab));
id2word_glove = containers.Map(1:length(word2vec_vocab), word2vec_vocab);

%%  Create structure 

Sent = {};  % struct used by Karpathy
k = 1;  % index over the new struct Sent
for j = 1:size(csv_table,1)  % iterate over rows in csv table

    % initialize a structure for each word
    Sent_item = struct('sStr',[], 'sNums', {}, 'img_id', [], 'img_filename', {}, 'folder', {});
    
    % sNums: is an array that contains the index (or indices) of the word2vec vocabulary that 
    %corresponds to the word(s).
    % sStr: sentence or word

    word_index = -1;  % -1 by default if not found in the word2vec vocabulary
    
    word = csv_table.('word')(j);  % get the word field
    % if the word occurs both in the dress vocab and in the word2vec vocab,
    %, update word_index to the word2vec id that corresponds to the word
    if (isKey(word2id_dress,word{1})) && (isKey(word2id_glove,word{1}))
        word_index = word2id_glove(word{1});
    
        Sent_item(1).sNums = word_index;
        Sent_item(1).sStr = word;
        Sent_item(1).img_id = csv_table.('img_id')(j);
        Sent_item(1).img_filename = csv_table.('img_filename')(j);
        Sent_item(1).folder = csv_table.('folder')(j);
        
        Sent{k} = Sent_item;
        k = k + 1;
            
    end

end

%%
% save matlab struct
fname = struct_info.out_mat_fname;
save(fname, 'Sent', '-v7.3')  % With the -v7.3 flag you can store variables > 2GB with compression
% http://www.mathworks.com/matlabcentral/answers/15521-matlab-function-save-and-v7-3




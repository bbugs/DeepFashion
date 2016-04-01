%%%%  
% Load csv files with all sentences and make the structures to input into 
% Karpathy's code
%%%%
% specify desired split
clear all;
which_split = 'val';
with_zappos = 'no_zappos';  

%% Load csv file with sentences per image

fname = sprintf('../../data/fashion53k/csv/fashion53k_%s.csv', with_zappos);


fid = fopen(fname, 'rt');

% str = '0,0,train,B0009PDO0Y,dress_attributes/data/images/BridesmaidDresses/,satin halter dress prom gown bridesmaid';

% img_id	sent_id	split	asin	folder	sentence
% out=textscan(str,'%s%s%q', 'delimiter',',')
C = textscan(fid, '%s%s%s%s%s%s', 'HeaderLines', 1, 'CollectOutput', true, 'delimiter', ',');
fclose(fid);

img_id = C{1,1}(:,1);
sent_id = C{1,1}(:,2);
split = C{1,1}(:,3);
asin = C{1,1}(:,4);
folder = C{1,1}(:,5);
sentences = C{1,1}(:,6);

%% Load dress vocabulary
% fname = '../../data/fashion53k/vocab/vocab_no_zappos.txt';
fname = sprintf('../../data/fashion53k/vocab/vocab_%s.txt', with_zappos);
fid = fopen(fname, 'rt');
C = textscan(fid, '%s', 'HeaderLines', 0, 'CollectOutput', true);
fclose(fid);
dress_vocab = C{1};


%% Load glove or word2vec vocabulary
fname = '../../data/word_vects/vocab.txt';
fid = fopen(fname, 'rt');
C = textscan(fid, '%s', 'HeaderLines', 0, 'CollectOutput', true);
fclose(fid);
glove_vocab = C{1};

%% Create word2id_dress and id2word_dress and word2id_glove and id2word_glove

word2id_dress = containers.Map(dress_vocab,1:length(dress_vocab));
% word2id_dress('satin')
id2word_dress = containers.Map(1:length(dress_vocab), dress_vocab);

%%
word2id_glove = containers.Map(glove_vocab, 1:length(glove_vocab));
id2word_glove = containers.Map(1:length(glove_vocab), glove_vocab);

%%  Create structure 

Sent = {};
j = 1;  % index over all sentences
k = 1;  % index over sentences in the specific split
for j = 1:length(sentences)
    
    if strcmp(split{j}, which_split)  % compare strings
        % initialize a structure for each sentence
        s = struct('sStr',[], 'sNums', {}, 'img_id', [], 'sent_id', [], 'asin', {}, 'split', {}, 'folder', {}, 'txt', '');

        words = regexp(sentences{j},'\s+','split');  % split sentence into words

        tmp = -ones(1, length(words));  % initialize sNums to -1
        i = 1;
        for word = words
        %  fprintf('%s\n',word{1});

          if (isKey(word2id_dress,word{1})) && (isKey(word2id_glove,word{1}))
            tmp(i) = word2id_glove(word{1});
          end

          i = i + 1;
        end
        s(1).sNums = tmp;
        s(1).sStr = words;
        s(1).img_id = img_id{j};
        s(1).sent_id = sent_id{j};
        s(1).asin = asin{j};
        s(1).split = split{j};
        s(1).folder = folder{j};
        s(1).txt = sentences{j};

        Sent{k} = s;
        k = k + 1;
    end
    
end

fname = sprintf('../../data/fashion53k/matlab_structs/no_zappos/split_%s_sent.mat', which_split);
save(fname, 'Sent')


%% Load vocabulary from glove or word2vec

% vocab = readTextFile('../../data/word_vects/vocab.txt')';
% oWe = load('../../data/word_vects/vocab_vecs.txt')';
% save('common/wordvecs_200d_word2vec.mat', 'vocab', 'oWe');
% 
% word_dim = 200;
% fsave = sprintf('../../data/word_vects/vectors_%dd.mat', word_dim);
% 
% load(fsave, 'oWe', 'vocab');

%  
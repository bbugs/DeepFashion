"""
Read in interim files to create files that can be fed to the lstm ptb_fashion model
"""

# read in the vocabulary

# and keep the words that happen more than 5 times

# keep the vocabulary.

with_zappos = 'no_zappos'  # no_zappos, with_ngrams, only_zappos
rpath = '/Users/susanaparis/Documents/Belgium/DeepFashion/data/fashion53k/text/'

import collections

##  Read the train data
split = 'train'
fname = rpath + '/concat_interim/{}/fashion53k.{}.txt'.format(with_zappos, split)
with open(fname, 'r') as f:
    text = f.read()

# text = text.replace('\n', " <eos> ")

sentences = text.split('\n')

# word list
data = text.split()
counter = collections.Counter(data)
count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
top_count_pairs = [pair for pair in count_pairs if pair[1] > 5]  # keep words that occur more than 5 fimes

vocabulary_words, _ = list(zip(*top_count_pairs))
print "vocabulary size", len(vocabulary_words), with_zappos


#####  Write the train data  ####
fout_name = rpath + 'text_data/{}/fashion53k.{}.txt'.format(with_zappos, split)
print "writing to file train"
f = open(fout_name, 'w')

i = 0
for sentence in sentences:
    if i % 10000 == 0:
        print i
    new_sentence = [w if w in vocabulary_words else '<unk>' for w in sentence.split()]
    new_text = '<sos> ' + ' '.join(new_sentence) + '\n'
    f.write(new_text)
    i += 1
f.close()


# word_to_id = dict(zip(words, range(len(words))))

####  Validation data ####
# Read in the validation text and project the text into this vocabulary
split = 'val'
f_val_name = rpath + '/concat_interim/{}/fashion53k.{}.txt'.format(with_zappos, split)
with open(f_val_name, 'r') as f:
    text = f.read()
sentences = text.split('\n')

fout_name = rpath + '/text_data/{}/fashion53k.{}.txt'.format(with_zappos, split)
print "writing to file validation"
f = open(fout_name, 'w')
i = 0
for sentence in sentences:
    if i % 10000 == 0:
        print i
    new_sentence = [w if w in vocabulary_words else '<unk>' for w in sentence.split()]
    # [unicode(x.strip()) if x is not None else '' for x in row]
    new_text = '<sos> ' + ' '.join(new_sentence) + '\n'
    f.write(new_text)
    i += 1
f.close()


####  Test data ####
# Read in the validation text and project the text into this vocabulary
split = 'test'
fname = rpath + '/concat_interim/{}/fashion53k.{}.txt'.format(with_zappos, split)
with open(fname, 'r') as f:
    text = f.read()
sentences = text.split('\n')

fout_name = rpath + 'text_data/{}/fashion53k.{}.txt'.format(with_zappos, split)
print "writing to file test"
f = open(fout_name, 'w')
i = 0
for sentence in sentences:
    if i % 10000 == 0:
        print i
    new_sentence = [w if w in vocabulary_words else '<unk>' for w in sentence.split()]
    # [unicode(x.strip()) if x is not None else '' for x in row]
    new_text = '<sos> ' + ' '.join(new_sentence) + '\n'
    f.write(new_text)
    i += 1
f.close()



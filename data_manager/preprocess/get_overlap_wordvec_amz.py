"""
See how much overlap there is between fashion53k and the wordvec from glove or word2vec
"""
import csv
import sys

# read dress vocab
with_zappos = 'no_zappos'
fname = '../../../../data/fashion53k/vocab/vocab_{}.txt'.format(with_zappos)

csvfile = open(fname, 'rb')

reader = csv.reader(csvfile)
word_list = []
for row in reader:
    word_list.append(row[0])

# print word_list[0:10]
csvfile.close()

print "dress vocab size ", len(word_list)


# read glove vocab
fname = '../../../../data/word_vects/vocab.txt'
csv.field_size_limit(sys.maxsize)
f = open(fname, 'rb')

lines = f.readlines()
glove_words = [w.strip() for w in lines]

print "glove vocab size ", len(glove_words)

# print glove_words[0:10]

common_words = set(word_list).intersection(glove_words)

print "number of common words ", len(common_words)

not_common = set(word_list).difference(set(glove_words))

print "number of non common words ", len(not_common)

print not_common

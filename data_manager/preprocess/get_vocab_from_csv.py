"""
Get all text from csv files and extract the vocabulary

The csv files have the following columns
['img_id', 'sent_id', 'split', 'asin', 'folder', 'sentence']



"""


import pandas as pd
import collections

# Read in csv file and write vocabulary

with_zappos = 'no_zappos'
# with_zappos = 'with_ngrams' # only_zappos is not needed because it already exists from zappos

fname = '../../../../data/fashion53k/csv/fashion53k_{}.csv'.format(with_zappos)


sentences_df = pd.read_csv(fname)
# print sentences_df[0:10]
sentences_df.columns = ['img_id', 'sent_id', 'split', 'asin', 'folder', 'sentence']

# print sentences_df.columns
text = ''
for item in sentences_df['sentence']:
    text += item + ' '

# print text
data = text.split()
counter = collections.Counter(data)
count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
top_count_pairs = [pair for pair in count_pairs if pair[1] > 5]  # keep words that occur more than 5 fimes


vocabulary_words, _ = list(zip(*top_count_pairs))

fname = '../../../../data/fashion53k/vocab/vocab_{}.txt'.format(with_zappos)

fout = open(fname, 'wb')


for item in vocabulary_words:
    print>>fout, item

fout.close()



# print "vocabulary size", len(vocabulary_words), with_zappos
#
# print vocabulary_words


# csvfile = open('eggs.csv', 'rb')
# csvreader = csv.reader(csvfile)
#
# for row in csvreader:
#     content = list(row[i] for i in included_cols)
#     print content


import collections





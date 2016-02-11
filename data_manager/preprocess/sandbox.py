
import json
import re
from nltk.tokenize import MWETokenizer
from nltk import word_tokenize
import nltk
from zappos_ngrams import zappos_ngrams

# read json file
rpath = '/Users/susanaparis/Documents/Belgium/IMAGES_plus_TEXT/DATASETS/dress_attributes/data/json/'
fname = rpath + 'dataset_dress_all_train_val.json'

data = json.load(open(fname, 'rb'))

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer

word_tokenizer = RegexpTokenizer(r'\w+')

# multiple word tokenizer
tokenizer_mwe = MWETokenizer(zappos_ngrams)  # all zappos_ngrams are added

# concatenate the text from all the items
i = 0
text = ''
for item in data['items'][0:1000]:
    print i
    i += 1
    t = item['text']
    # substitute word1.word2 by word1. word2
    t = re.sub(r"(\w[A-Z]|[a-z.])\.([^.)\s])", r"\1. \2", t)
    rough_sentences = t.split('<\\%>')

    sentences = []
    for rs in rough_sentences:
        sentences.extend(sent_tokenizer.tokenize(rs))

    for sentence in sentences:
        words = word_tokenizer.tokenize(sentence.lower())  # tokenize based on words
        words = tokenizer_mwe.tokenize(words)  # group zappos_ngrams into one token.
        words_concat = ' '.join(words) + '\n '
        text += words_concat


with open('test2.txt', 'w') as f:
    f.write(text)

"""
Concatenate all the text from all the items in the json file and write it to an interim file
"""
import argparse
import json
import re
import nltk
from zappos_vocabulary import ZAPPOS_VOCAB_LIST, ZAPPOS_VOCAB_TUPLES
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer  # multiple word tokenizer to process zappos ngrams


SENT_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')  # tokenizer based on punctuation
WORD_TOKENIZER = RegexpTokenizer(r'\w+')
# multiple word tokenizer
TOKENIZER_MWE = MWETokenizer(ZAPPOS_VOCAB_TUPLES)  # all zappos_words are added


def clean_text(dirty_text, with_zappos):
    """(str) -> str


    The text is cleaned in the following way:
    # substitute word1.word2 by word1. word2
    # split text into rough sentences based on '<\\%>'.  This symbol was added to denote
      a new line on the original product description
    # split the rough sentences using a sentence tokenizer from nltk
    # if zappos_ngrams is True, combine the zappos ngrams into one token. e.g., short sleeves -> short_sleeves

    # concatenate all tokenized words into one string and return string.

    An excerpt of text from the item looks like this:
    Sheath/Column One-Shoulder Short/Mini Bridesmaid Dress With Lace<\\%>SKU#:1020234<\\%>New Style Cocktail Dress<\\%>
    Color:The color as picture<\\%>Category:Bridesmaid Dress<\\%> Brand:Biggoldapple<\\%>
    Silhouette:Sheath/Column<\\%> Neckline:One-Shoulder<\\%> Hemline/Train:Short/Mini<\\%>

    """
    out_clean_text = ''
    # substitute word1.word2 by word1. word2
    dirty_text = re.sub(r"(\w[A-Z]|[a-z.])\.([^.)\s])", r"\1. \2", dirty_text)
    rough_sentences = dirty_text.split('<\\%>')  # sentences based on splitting by '<\\%>'

    sentences = []
    for rs in rough_sentences:
        sentences.extend(SENT_TOKENIZER.tokenize(rs))  # sentences based on NLTK tokenizer

    for sentence in sentences:
        words = WORD_TOKENIZER.tokenize(sentence.lower())  # tokenize based on words. ignore that zappos vocabulary exists

        if with_zappos == 'with_ngrams':
            # keep all words (even those not in zappos), but group zappos ngrams into one token
            words = TOKENIZER_MWE.tokenize(words)  # group zappos_ngrams into one token.
        elif with_zappos == 'only':
            words = TOKENIZER_MWE.tokenize(words)  # group zappos_ngrams into one token.
            words = [w for w in words if w in ZAPPOS_VOCAB_LIST]  # only keep words in the zappos vocabulary

        words_concat = ' '.join(words) + '\n'
        out_clean_text += words_concat

    return out_clean_text


def get_sentences_from_item(item, with_zappos):
    """(dict, bool) -> list
    Given an item, return the clean text.
    """

    item_text = item['text']

    concat_clean_text = clean_text(item_text, with_zappos)
    sentence_list = concat_clean_text.split('\n')

    return sentence_list


class FoutConfig(object):
    def __init__(self, dir_out, with_zappos, split):
        if with_zappos == 'no':
            fout = dir_out + '/no_zappos/fashion53k.{}.txt'.format(split)
        elif with_zappos == 'with_ngrams':
            fout = dir_out + '/with_ngrams/fashion53k.{}.txt'.format(split)
        elif with_zappos == 'only':
            fout = dir_out + '/only_zappos/fashion53k.{}.txt'.format(split)
        else:
            raise ValueError("--zappos must be either no, with_ngrams, or only")
        print "file out name", fout
        self.fout = open(fout, 'w')
        return


def main(params):

    with_zappos = params['zappos']
    fname_in = params['data_in_fname']
    data = json.load(open(fname_in, 'rb'))
    split = params['split']
    dir_out = params['data_out_directory']

    # clean and concatenate the text from all the items
    i = 0
    text = ''
    file_out_handle = FoutConfig(dir_out, with_zappos, split).fout
    print "processing json file"
    for item in data['items']:
        if i % 10000 == 0:
            print "item", i
        if item['split'] == split:
            sentence_list = get_sentences_from_item(item, with_zappos=with_zappos)
            for l in sentence_list:
                if len(l) == 0:
                    continue
                if not l.strip().isspace():
                    file_out_handle.write(l + '\n')
            # text  += t
        i += 1

    file_out_handle.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    rpath = '/Users/susanaparis/Documents/Belgium/'

    parser.add_argument('-s', '--split', dest='split', type=str,
                        default='train',
                        help='split options: "train", "val", "test"')

    # specify if you want to use zappos
    parser.add_argument('-z', '--zappos', dest='zappos', type=str, default='only',
                        help='zappos: options: "no", "with_ngrams", "only"')

    parser.add_argument('--data_in_fname', dest='data_in_fname', type=str,
                        default=rpath + '/IMAGES_plus_TEXT/DATASETS/dress_attributes/data/json/dataset_dress_all.json',
                        help='path to json file')

    parser.add_argument('--data_out_directory', dest='data_out_directory', type=str,
                        default='../../../../data/fashion53k/text/concat_interim/',
                        help='zappos: options: "no", "with_ngrams", "only"')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print 'parsed parameters:'
    print json.dumps(params, indent=2)
    main(params)

    # example call
    # python concat_all_text_interim.py -s train -z no

"""
Concatenate all the text from all the items in the json file and write it to an interim file
"""
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
        elif with_zappos == 'only_zappos':
            words = TOKENIZER_MWE.tokenize(words)  # group zappos_ngrams into one token.
            words = [w for w in words if w in ZAPPOS_VOCAB_LIST]  # only keep words in the zappos vocabulary

        words = [w for w in words if (not w.isdigit() or w == '3/4')]  # remove words that are just digits, but leave 3/4

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



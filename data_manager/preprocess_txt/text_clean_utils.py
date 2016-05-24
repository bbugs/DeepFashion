"""
Concatenate all the text from all the items in the json file and write it to an interim file
"""
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import MWETokenizer  # multiple word tokenizer to process zappos ngrams

SENT_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')  # tokenizer based on punctuation
WORD_TOKENIZER = RegexpTokenizer(r'\w+')


def load_vocab_words(vocab_filename):
    with open(vocab_filename, 'r') as f:
        vocab_words = [w.strip() for w in f.readlines()]
    return vocab_words


def clean_text(dirty_text, external_vocab_filename=None, external_vocab_level="no"):
    """(str, str, str) -> str

    external_vocab_level can be: no, with_ngrams, only.
    if you choose with_ngrams or only, you need to add an external_vocab_filename

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

    external_vocab_words = []
    if external_vocab_filename is not None:
        external_vocab_words = load_vocab_words(external_vocab_filename)

        # transform ngrams into tuples
        external_vocab_with_tuples = [tuple(z.split('_')) for z in external_vocab_words]  # assume that ngrams are separated by underscore: word1_word2.
        # multiple word tokenizer, more info: http://www.nltk.org/api/nltk.tokenize.html
        tokenizer_mwe = MWETokenizer(external_vocab_with_tuples)  # all external_vocab_words are added

    out_clean_text = ''
    # substitute word1.word2 by word1. word2
    dirty_text = re.sub(r"(\w[A-Z]|[a-z.])\.([^.)\s])", r"\1. \2", dirty_text)
    rough_sentences = dirty_text.split('<\\%>')  # sentences based on splitting by '<\\%>'

    sentences = []
    for rs in rough_sentences:
        rs = rs.replace("3/4", "3_4")  # just to keep the 3/4 as 3_4
        sentences.extend(SENT_TOKENIZER.tokenize(rs))  # sentences based on NLTK tokenizer

    for sentence in sentences:
        words = WORD_TOKENIZER.tokenize(sentence.lower())  # tokenize based on words. ignore that zappos vocabulary exists

        if external_vocab_level == 'with_ngrams':
            # keep all words (even those not in zappos), but group zappos ngrams into one token
            words = tokenizer_mwe.tokenize(words)  # group zappos_ngrams into one token.
        elif external_vocab_level == 'only':
            words = tokenizer_mwe.tokenize(words)  # group zappos_ngrams into one token.
            words = [w for w in words if w in external_vocab_words]  # only keep words in the zappos vocabulary

        words = [w for w in words if (not w.isdigit() or w == '3_4')]  # remove words that are just digits, but leave 3_4

        words_concat = ' '.join(words) + '\n'
        out_clean_text += words_concat

    return out_clean_text


def get_sentences_from_item(item, external_vocab_filename, external_vocab_level):
    """(dict, str, str) -> list
    Given an item, return the clean text possibly using an external vocabulary
    """

    item_text = item['text']

    concat_clean_text = clean_text(item_text, external_vocab_filename, external_vocab_level)
    sentence_list = concat_clean_text.split('\n')

    return sentence_list


if __name__ == "__main__":

    # dirty_text_ex = """Red Yellow Green Tie Dye Slinky Print V Neck Tank Plus Size
    # Supersize Maxi Dress<\\%>Standard length (50\"), Tall length (54\")<\\%>0x:
    # Chest:48\" - Hips (appx):58\" 1x: Chest:52\" - Hips (appx):62\" 2x: Chest:56\" -
    # Hips (appx):66\" 3x: Chest:60\" - Hips (appx):70\" 4x: Chest:64\" - Hips (appx):74\" 5x:
    # Chest:68\" - Hips (appx):78\" 6x: Chest:72\" - Hips (appx):82\" 7x: Chest:76\" - Hips (appx):86\"
    # 8x: Chest:80\" - Hips (appx):89\" 9x: Chest:84\" - Hips (appx):92\"<\\%>Standard A-Line Maxi Dress,
    # polyester/spandex blend<\\%>Tank (Sleeveless), V Neckline<\\%>OUR BRAND RUNS VERY BIG!!<\\%>
    # Standard A-Line Cut Slinky Tank Dress With V Neckline.<\\%>The fabric is a lightweight polyester
    # spandex blend.<\\%>It is wonderful and a breeze to take care of!<\\%>Throw it in the washer,
    # dry flat, and then straight into a drawer or suitcase, take it out days or weeks later, and it's
    # ready to wear!<\\%>Please note that this material is very stretchy.<\\%>They run VERY big and are
    # extra roomy!<\\%>Compare your measurements to the size chart to accurately place your size to order."""

    dirty_text_ex = "3/4 sleeves"

    print clean_text(dirty_text_ex, external_vocab_filename="../../data/fashion53k/vocab/zappos.vocab.txt",
                     external_vocab_level="no")

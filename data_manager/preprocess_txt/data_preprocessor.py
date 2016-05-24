"""

Given a json file with ['items] and a ['text] field for each item, you can
use this function to create a file with an rnn format

"""

from data_manager.data_provider import DataProvider


def json2rnn_format(in_json_fname, vocabulary_words, out_rnn_fname=None):
    """(str, list, str, str) -> save rnn file in out_rnn_fname
    Write each line.
    out_filename where to save the file for rnn processing.
    The out rnn file is inspired by the ptb data from tensorflow

    """

    dp = DataProvider(dataset_fname=in_json_fname)

    all_text = dp.get_all_txt_from_json()

    sentences = all_text.split('\n')

    f = open(out_rnn_fname, 'w')
    i = 0
    for sentence in sentences:
        if i % 10000 == 0:
            print i
        new_sentence = [w if w in vocabulary_words else '<unk>' for w in sentence.split()]
        new_text = '<sos> ' + ' '.join(new_sentence) + ' \n'  # note: <eos> token gets added by the PTB reader.
        f.write(new_text)
        i += 1
    f.close()

    return


# class DataPreProcessor(object):
#
#     def __init__(self, train_json_fname=None, val_json_fname=None, test_json_fname=None):
#         if train_json_fname is not None:
#             self.train_data = DataProvider(train_json_fname)
#         if val_json_fname is not None:
#             self.val_data = DataProvider(val_json_fname)
#         if test_json_fname is not None:
#             self.test_data = DataProvider(test_json_fname)
#
#
#     def json2rnn_format(self, json_fname, vocabulary_words, out_filename=None):
#         """
#         Write each line.
#         Keep only words above min_word_freq
#         out_filename where to save the file for rnn processing
#         """
#
#         train_vocabulary = ""
#
#         pass
#
#     def create_matlab_struct_txt_4karpathy(self):
#         """
#         Call matlab code
#         """
#         pass

if __name__ == "__main__":

    # Extract vocabulary words from train json file
    min_word_freq = 5
    dp = DataProvider(dataset_fname="../../data/fashion53k/json/no_zappos/dataset_dress_all_train.clean.json")  # always should be the train split here to read the vocabulary.
    vocab_words = dp.get_vocab_words_from_json(min_word_freq=min_word_freq)

    # create rnn file for each split and zappos conditions
    split = "test"  # train, val, test
    with_zappos = "with_ngrams"  # no_zappos, with_ngrams, only_zappos
    json_fname = "../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json".format(with_zappos, split)

    rnn_fname = "../../data/fashion53k/text_4_rnn/{}/{}.rnn.txt".format(with_zappos, split)
    json2rnn_format(json_fname, vocab_words, rnn_fname)

    pass
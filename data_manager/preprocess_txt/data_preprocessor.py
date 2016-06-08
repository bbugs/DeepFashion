"""

Given a json file with ['items] and a ['text] field for each item, you can
use this function to create a file with an rnn format

"""

from data_manager.data_provider import DataProvider
import os
import numpy as np
import scipy.io

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


def json2matlab_struct_txt(in_json_fname, out_mat_fname="", vocabulary=None, per_word=True,
                           word2vec_vocab_fname="", word2vec_fname="", delete=True):
    """(str, str, list or set, boolean, str, str, bool)

    convert in json file to a matlab struct to be used in karpathy's matlab code
    """

    # Read json file into a DataProvider object
    dp = DataProvider(dataset_fname=in_json_fname)

    # create a temporary vocabulary file
    fashion_vocab_fname = "tmp.vocab.txt"
    with open(fashion_vocab_fname, "w") as f:
        for w in vocabulary:
            f.write("{}\n".format(w))

    # Create a temporary csv file with fields. If per_word is True, then write one word per line.
    # If not, write one sentence per line
    csv_fname = "tmp.csv"
    fieldnames = ['img_id', 'split', 'folder', 'img_filename',
                  'asin', 'sentence', 'word']
    if per_word:
        fieldnames.remove('sentence')
    else:
        fieldnames.remove('word')
    dp.json2csv(fieldnames, fout_name=csv_fname, vocabulary=vocabulary)

    # create struct_info and save it so matlab can read it and extract meta information for the text structure
    fashion_data = {
        'struct_info': {
            'csv_fname': csv_fname,  # tmp.csv
            'fashion_vocab_fname': fashion_vocab_fname,  # tmp.vocab.txt
            'word2vec_vocab_fname': word2vec_vocab_fname,  # ../../data/word_vects/glove/vocab.txt
            'word2vec_fname': word2vec_fname,  # ../../data/word_vects/glove/vectors_200d.txt
            'fieldnames': np.array(fieldnames, dtype=np.object),
            'out_mat_fname': out_mat_fname,  # filename of matlab structure for karpathy
        }
    }

    struct_info_fname = out_mat_fname.replace(".mat", ".meta.mat")
    scipy.io.savemat(struct_info_fname, fashion_data)  # save the meta info
    scipy.io.savemat("struct_info.tmp.mat", fashion_data)  # save a temp of the meta info

    # call matlab
    if per_word:
        os.system("matlab -nojvm -nodesktop -nosplash < data_manager/preprocess_txt/clients/create_txt_struct_per_word.m")
    else:
        raise NotImplementedError("TODO later in case you want each sentence "
                                  "instead of each word, write matlab code for it")

    if delete:
        os.remove(fashion_vocab_fname)
        os.remove(csv_fname)
        os.remove("struct_info.tmp.mat")

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

    # See clients for example run create_txt_structs.py

    # Load vocabulary
    # dp_train = DataProvider(dataset_fname="../../data/fashion53k/json/no_zappos/dataset_dress_all_test.clean.json")
    # vocab_train = dp_train.get_vocab_words_from_json(min_word_freq=5)
    #
    # # json to transfrom to matlab struct
    # json_fname ="../../data/fashion53k/json/no_zappos/dataset_dress_all_val.clean.json"
    #
    # json2matlab_struct_txt(json_fname, out_mat_fname="val.test.word.mat", vocabulary=vocab_train, per_word=True,
    #                        word2vec_vocab_fname="../../data/word_vects/glove/vocab.txt",
    #                        word2vec_fname="../../data/word_vects/glove/vectors_200d.txt", delete=False)


    pass
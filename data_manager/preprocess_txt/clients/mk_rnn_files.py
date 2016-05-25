
from data_manager.data_provider import DataProvider
from data_manager.preprocess_txt import data_preprocessor

# Extract vocabulary words from train json file
min_word_freq = 5
with_zappos = "only_zappos"  # no_zappos, with_ngrams, only_zappos
train_json_fname = "../../data/fashion53k/json/{}/dataset_dress_all_train.clean.json".format(with_zappos)
dp = DataProvider(dataset_fname=train_json_fname)  # always should be the train split here to read the vocabulary.
vocab_words = dp.get_vocab_words_from_json(min_word_freq=min_word_freq)

# create rnn file for each split and zappos conditions
split = "test"  # train, val, test
json_fname = "../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json".format(with_zappos, split)
rnn_fname = "../../data/fashion53k/text_4_rnn/{}/{}.rnn.txt".format(with_zappos, split)
data_preprocessor.json2rnn_format(json_fname, vocab_words, rnn_fname)


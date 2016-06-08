from data_manager.preprocess_txt import data_preprocessor
from data_manager.data_provider import DataProvider

# Load vocabulary
with_zappos = 'no_zappos'
fname = "../../data/fashion53k/json/{}/dataset_dress_all_train.clean.json".format(with_zappos)
dp_train = DataProvider(dataset_fname=fname)
vocab_train = dp_train.get_vocab_words_from_json(min_word_freq=5)

# json to transfrom to matlab struct
split = 'test'
json_fname = "../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json".format(with_zappos, split)

out_matlab_fname = "../../data/fashion53k/matlab_structs/per_word/text/{}.{}.mat".format(split, with_zappos)
data_preprocessor.json2matlab_struct_txt(json_fname, out_mat_fname=out_matlab_fname,
                                         vocabulary=vocab_train, per_word=True,
                                         word2vec_vocab_fname="../../data/word_vects/glove/vocab.txt",
                                         word2vec_fname="../../data/word_vects/glove/vectors_200d.txt", delete=False)





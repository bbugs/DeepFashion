
from data_manager.preprocess_txt.data_cleaner import DataCleaner

########################################################################################
##  no Zappos
########################################################################################

with_zappos = "no_zappos"

split = "test"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=None,
                   external_vocab_level="no",
                   out_filename=out_filename)

split = "val"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=None,
                   external_vocab_level="no",
                   out_filename=out_filename)

split = "train"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=None,
                   external_vocab_level="no",
                   out_filename=out_filename)

########################################################################################
##  With Zappos ngrams
########################################################################################

with_zappos = "with_ngrams"
external_vocab_fname = "../../data/fashion53k/external_vocab/zappos.vocab.txt"

split = "test"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=external_vocab_fname,
                   external_vocab_level="with_ngrams",
                   out_filename=out_filename)

split = "val"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=external_vocab_fname,
                   external_vocab_level="with_ngrams",
                   out_filename=out_filename)

split = "train"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=external_vocab_fname,
                   external_vocab_level="with_ngrams",
                   out_filename=out_filename)

########################################################################################
##  With zappos vocabulary only
########################################################################################

with_zappos = "only_zappos"

split = "test"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=external_vocab_fname,
                   external_vocab_level="only",
                   out_filename=out_filename)

split = "val"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=external_vocab_fname,
                   external_vocab_level="only",
                   out_filename=out_filename)

split = "train"
dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
dc = DataCleaner(dataset_fname=dataset_fname)
dc.json2clean_json(target_split=split,
                   external_vocab_filename=external_vocab_fname,
                   external_vocab_level="only",
                   out_filename=out_filename)
"""
Create new json files with clean text and the option to use an external vocabulary (e.g., zappos)

"""
import json
from text_clean_utils import get_sentences_from_item
from utils_local import utils_local


class DataCleaner(object):
    """

    """

    def __init__(self, dataset_fname='dataset/dataset_dress_all.json'):
        """
        """
        self.dataset = utils_local.load_data0(dataset_fname)

        return

    def json2clean_json(self, target_split="train",
                        external_vocab_filename=None,
                        external_vocab_level="with_ngrams",
                        out_filename=""):
        """
        from the json file containing the raw dataset, create a new json file after cleaning the text
        The cleaning is specified in clean_text() method in pp_utils.
        It is also possible to specify whether the new json file should become aware of an external vocabulary.
        If the zappos vocabulary is not desired, then use externa_vocab_level = "no" and external_vocab_filename=None
        """

        data = self.dataset

        new_data = {}
        new_data['items'] = []
        i = 1
        for item in data['items']:

            if i % 10000 == 0:
                print "item", i
            new_text = ''
            if item['split'] == target_split:
                sentence_list = get_sentences_from_item(item, external_vocab_filename, external_vocab_level)
                for l in sentence_list:
                    if len(l) == 0:
                        continue
                    if not l.strip().isspace():
                        new_text += l + '\n '
                        # text  += t
                item['text'] = new_text

            i += 1

            new_data['items'].append(item)

        with open(out_filename, 'w') as file_handle:
            json.dump(new_data, file_handle, indent=4)

        return


if __name__ == '__main__':
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


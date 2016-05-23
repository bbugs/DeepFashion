"""


"""
import json
from pp_utils import get_sentences_from_item


from utils_local import utils_local


class DataCleaner(object):
    """

    """

    def __init__(self, dataset_fname='dataset/dataset_dress_all.json'):
        """
        """
        self.dataset = utils_local.load_data0(dataset_fname)

        return

    def json2clean_json(self, target_split="train", with_zappos="no", out_filename=""):
        """
        from the json file containing the raw dataset, create a new json file, where some cleaning has been done.
        The cleaning is specified in clean_text() method in pp_utils.
        It is also possible to specify whether the new json file should become aware of the zappos vocabulary.
        If the zappos vocabulary is not desired, then use with_zappos = "no".  Right now, we only support the zappos
        vocabulary but later on we can support any external vocabulary.
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
                sentence_list = get_sentences_from_item(item, with_zappos=with_zappos)
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

    split ="test"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)

    split = "val"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)

    split = "train"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)

    ########################################################################################
    ##  With Zappos ngrams
    ########################################################################################

    with_zappos = "with_ngrams"

    split = "test"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)

    split = "val"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)

    split = "train"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)

    ########################################################################################
    ##  With Zappos ngrams
    ########################################################################################

    with_zappos = "only_zappos"

    split = "test"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)

    split = "val"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)

    split = "train"
    dataset_fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(split)
    out_filename = '../../data/fashion53k/json/{}/dataset_dress_all_{}.clean.json'.format(with_zappos, split)
    dc = DataCleaner(dataset_fname=dataset_fname)
    dc.json2clean_json(target_split=split, with_zappos=with_zappos,
                       out_filename=out_filename)
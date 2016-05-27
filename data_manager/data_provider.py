
import json
import csv
import os
import collections


def get_vocabulary_words_with_counts(txt, min_word_freq):
    """(str, int) -> list
    Extract the vocabulary from a string that occur more than min_word_freq.
    Return a list of the vocabulary and the frequencies.
    """

    data = txt.split()
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # keep words that occur more than min_word_freq
    top_count_pairs = [pair for pair in count_pairs if pair[1] > min_word_freq]
    return top_count_pairs


class DataStats(object):

    def __init__(self, item_type='dresses',
                 dataset_fname='dataset/dataset_joint_all.json',
                 img_path_root='../'):
        """
        Load data
        item_type can be dresses,
        """
        with open(dataset_fname, 'r') as f:
            self.dataset = json.load(f)

        self.item_type = item_type


    def get_num_items(self):
        # eg. dataset['dresses']
        return len(self.dataset[self.item_type])


    def get_files_split(self, split_name='train'):
        split_filepaths = []
        for item in self.dataset[self.item_type]:
            if item['split'] == split_name:
                split_filepaths.append()


class DataProvider(object):
    """

    """

    def __init__(self, dataset_fname='dataset/dataset_dress_all.json'):
        """
        """
        with open(dataset_fname, 'r') as f:
            self.dataset = json.load(f)

        return

    def get_img_paths(self, verbose=False):
        """
        """
        img_paths = []
        for item in self.dataset['items']:
            folder = item['folder']
            img_path = folder + item['img_filename']
            if verbose:
                print img_path
            img_paths.append(img_path)
        return img_paths

    def get_asins_split(self, target_split='train'):
        asins = []
        for item in self.dataset['items']:
            asin = item['asin']
            split = item['split']
            if split == target_split:
                asins.append(asin)
        return asins

    def get_ids_split(self, target_split='train'):
        ids = []
        for item in self.dataset['items']:
            imgid = item['imgid']
            split = item['split']
            if split == target_split:
                ids.append(imgid)
        return ids

    def get_asins(self):
        asins = []
        for item in self.dataset['items']:
            asin = item['asin']
            asins.append(asin)
        return asins

    def save_json_splits(self, splits, fout_name, save=False):
        """splits is a set

        """
        split_dataset = {}
        split_dataset['items'] = []
        for item in self.dataset['items']:
            split = item['split']
            if split in splits:
                split_dataset['items'].append(item)
        if save:
            with open(fout_name, 'wb') as fp:
                json.dump(split_dataset, fp, indent=4, sort_keys=True)

    def json2csv(self, fieldnames, fout_name, vocabulary=None, target_split=None):
        """(list, str, set, str) -> save csv file as fout_name
        Read json file corresponding to split and write a csv file with img info:

        if sentence is the fieldname, whole sentences.
        if words is in the fieldname, then use the individual words.

        """
        possible_fieldnames = {'img_id', 'split', 'folder', 'img_filename',
                               'asin', 'sentence', 'word'}

        for fieldname in fieldnames:
            if fieldname not in possible_fieldnames:
                raise ValueError("fieldnames must be in {}".format(possible_fieldnames))
        if 'sentence' in fieldnames and 'word' in fieldnames:
            raise ValueError("choose either sentence or word field")

        csvfile = open(fout_name, 'wb')
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()

        for item in self.dataset['items']:
            if target_split is not None:
                if item['split'] != target_split:
                    continue
            d = {}
            if 'img_id' in fieldnames:
                d['img_id'] = item['imgid']
            if 'split' in fieldnames:
                d['split'] = item['split']
            if 'folder' in fieldnames:
                d['folder'] = item['folder']
            if 'img_filename' in fieldnames:
                d['img_filename'] = item['img_filename']
            if 'asin' in fieldnames:
                d['asin'] = item['asin']

            if 'sentence' in fieldnames or 'word' in fieldnames:
                if 'word' in fieldnames:
                    sentences = item['text'].split('\n ')  # assume that '\n ' separates each sentence
                    # collect words in all sentences
                    words = []
                    for sentence in sentences:
                        if len(sentence) == 0:
                            continue
                        words.extend(sentence.split())  # split sentence into words

                    # remove repeated words and write one word per line
                    unique_words = set(words)

                    # if a vocabulary is given, filter words by vocabulary
                    if vocabulary is not None:
                        unique_words = [w for w in unique_words if w in vocabulary]

                    for word in unique_words:
                        d['word'] = word
                        writer.writerow(d)

                if 'sentence' in fieldnames:
                    sentences = item['text'].split('\n ')  # assume that '\n ' separates each sentence
                    for sentence in sentences:
                        if len(sentence) == 0:
                            continue

                        if vocabulary is not None:
                            sentence = " ".join([w for w in sentence.split() if w in vocabulary])

                        if not sentence.isspace() and len(sentence) > 0:
                            d['sentence'] = sentence
                            writer.writerow(d)

            else:  # neither sentence nor words are in the fieldnames, i.e., don't deal with text
                writer.writerow(d)

        return

    def json2txt_file(self, fout_name):
        """
        Read json file corresponding and write a txt file with
        all the text from the json file one line for each line.
        Assume that the text on json is already clean.
        Here we lose all the info of which product belongs to what,
        but this is useful when you just want to see all
        the text, like when training an LSTM based on text alone
        """

        f = open(fout_name, 'w')

        i = 0
        for item in self.dataset['items']:
            text = item['text']
            sentences = text.split('\n ')  # assume that sentences end with "\n "
            for l in sentences:
                if len(l) == 0:
                    continue
                if not l.strip().isspace():
                    f.write(l + '\n')
            i += 1

        return

    def get_all_txt_from_json(self):

        """
        Concatenate all text from json and return it.
        """

        self.json2txt_file("tmp.txt")  # save a temp file with all the text
        with open("tmp.txt", 'r') as f:
            txt = f.read()

        os.remove("tmp.txt")  # remove temp file

        return txt

    def get_vocab_words_from_json(self, min_word_freq=5):
        """
        Get vocab words from json sorted by freq
        """
        all_text = self.get_all_txt_from_json()
        vocab_with_counts = get_vocabulary_words_with_counts(all_text, min_word_freq)
        vocab_words = [w[0] for w in vocab_with_counts]
        return vocab_words

    def get_vocab_words_from_json_with_counts(self, min_word_freq=5):
        """
        Get vocab words from json sorted by freq
        """
        all_text = self.get_all_txt_from_json()
        return get_vocabulary_words_with_counts(all_text, min_word_freq)


if __name__ == '__main__':
    # d = DataStats()
    #
    # print "number of dresses is: ", d.get_num_dress()

    # root_path = '../../DATASETS/dress_attributes/data/json/'
    # fname = root_path + 'dataset_dress_all.json'
    # d = DataProvider(dataset_fname=fname)
    #
    # print len(d.dataset)

    # d.get_img_paths(verbose=True)  # use this in the command line to write to file


    # Save dataset splits in json files
    # splits = {'train', 'val'}  # set of splits
    # fout_name = root_path + 'dataset_dress_all_train_val.json'
    # d.save_json_splits(splits, fout_name, save=True)
    #
    # splits = {'test'}  # set of splits
    # fout_name = root_path + 'dataset_dress_all_test.json'
    # d.save_json_splits(splits, fout_name, save=True)

    # Create csv files with image info
    # root_path = '../../data/fashion53k/'
    # target_split = 'train'
    # fname = root_path + 'json/dataset_dress_all_{}.json'.format(target_split)
    # fout_name = root_path + 'csv/imgs/imgs_info_{}.csv'.format(target_split)
    # d = DataProvider(dataset_fname=fname)
    # d.json2csv_imgs(target_split, fout_name)

    # dp = DataProvider("../../data/fashion53k/json/no_zappos/dataset_dress_all_train.clean.json")
    # dp.json2txt_file(fout_name="tmp.txt")

    # Test the get_vocabulary_words_with_freq method
    # text = """remedios boutique v neck chiffon sheath bridesmaid evening dress w keyhole back\n back with zipper fully lined built in bra\n for our ladies dresses should play an important role in our life while presenting in any formal or informal occasions and even in daily life\n here our shop remedios boutique is aimed to provide various kinds of dresses for your choices\n whether for a bridal party or other special occasions such as quinceanera party prom party etc you can find our dresses for yourself in different silhouettes and styles mainly textured in satin chiffon satin chiffon etc and with a variety of colors\n """
    #
    # print get_vocabulary_words_with_freq(text, min_word_freq=3)

    # dp = DataProvider(dataset_fname="../../data/fashion53k/json/only_zappos/dataset_dress_all_test.clean.json")
    # text = dp.get_all_txt_from_json()
    # print get_vocabulary_words_with_counts(text, min_word_freq=5)


    # Test json2csv


    dp_train = DataProvider(dataset_fname="../../data/fashion53k/json/only_zappos/dataset_dress_all_test.clean.json")
    vocab_train = set(dp_train.get_vocab_words_from_json(min_word_freq=5))

    dp = DataProvider(dataset_fname="../../data/fashion53k/json/no_zappos/dataset_dress_all_val.clean.json")
    dp.json2csv(['img_id', 'split', 'folder', 'img_filename', 'asin', 'sentence'], "tmp2.csv", vocab_train)

    dp.json2csv(['img_id', 'split', 'folder', 'img_filename', 'asin', 'sentence'], "tmp_none_vocab.csv")
    pass



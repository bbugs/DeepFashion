
import json
from utils_local import utils_local
import numpy as np
import csv


class DataStats(object):

    def __init__(self, item_type='dresses',
                 dataset_fname='dataset/dataset_joint_all.json',
                 img_path_root='../'):
        """
        Load data
        item_type can be dresses,
        """
        self.dataset = utils_local.load_data0(dataset_fname)
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
        self.dataset = utils_local.load_data0(dataset_fname)

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

    def json2csv_imgs(self, target_split, fout_name):
        """
        Read json file corresponding to split and write a csv file with img info:

        img_id, split, folder, img_filename

        """
        fieldnames = ['img_id', 'split', 'folder', 'img_filename']
        csvfile = open(fout_name, 'wb')
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()

        for item in self.dataset['items']:
            d = {}
            d['split'] = item['split']
            if d['split'] == target_split:
                d['img_id'] = item['imgid']
                d['split'] = item['split']
                d['folder'] = item['folder']
                d['img_filename'] = item['img_filename']
                writer.writerow(d)
        csvfile.close()
        return


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
    root_path = '../../data/fashion53k/'
    target_split = 'train'
    fname = root_path + 'json/dataset_dress_all_{}.json'.format(target_split)
    fout_name = root_path + 'csv/imgs/imgs_info_{}.csv'.format(target_split)
    d = DataProvider(dataset_fname=fname)
    d.json2csv_imgs(target_split, fout_name)




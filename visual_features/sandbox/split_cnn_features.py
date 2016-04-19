"""
Instead of splitting, it ended up easier to regenerate separately for each split.  Reading the cnn file in python takes way too long, whereas in matlab it takes 11 min.
"""

import sys
import json
import numpy as np
sys.path.append('/export/home2/NoCsBack/hci/susana/current/DeepFashion/projects/dress_project/')
from utils_local.utils_local import savetxt_compact


print "loading json file"
rpath = '../../data/fashion53k/'
fname = rpath + 'json/dataset_dress_all.json'
with open(fname, 'r') as f:
    data = json.load(f)

print "loading cnn file"
fname = rpath + 'img_regions/4_regions_cnn/cnn_fc7.txt'
f = open(fname, "rb")
cnn = np.loadtxt(f, delimiter=",")
f.close()

print type(cnn)
print cnn.shape

n_regions = 4
cnn_dim = 4096
target_layer = 'fc7'


n_imgs_val = 4000
n_imgs_test = 1000
n_imgs_train = 53689 - n_imgs_test - n_imgs_val


target_splits = ['test', 'val', 'train']

for target_split in target_splits:

    if target_split == 'train':
        n_rows = n_imgs_train * (n_regions + 1)
        new_cnn = np.zeros((n_rows, cnn_dim))
    elif target_split == 'val':
        n_rows = n_imgs_val * (n_regions + 1)
        new_cnn = np.zeros((n_rows, cnn_dim))
    elif target_split == 'test':
        n_rows = n_imgs_test * (n_regions + 1)
        new_cnn = np.zeros((n_rows, cnn_dim))
    else:
        raise ValueError("must be either train, val, test")

    new_index = 0 # runs through the new cnn matrix
    for item in data['items']:
        img_id = item['imgid']
        split = item['split']
        asin = item['asin']
        folder = item['folder']
        img_filename = item['img_filename']
        # print folder, img_filename
        # if img_id % 1000 == 0:
        #     print img_id

        index0 = img_id
        # print "index0", index0

        if img_id % 1000 == 0:
            print target_split
            print "img_id", img_id

        if split == target_split:
            index1 = index0 + (n_regions + 1)
            # print "index1", index1
            cnn_slice = cnn[index0:index1, :]
            new_cnn[new_index:(new_index + n_regions + 1), :] = cnn_slice
            new_index += (n_regions + 1)
            print "new_index", new_index

    print "saving cnn features"
    cnn_out_path = '../../data/fashion53k/img_regions/{}_regions_cnn/per_split/'.format(n_regions)
    fout_name = cnn_out_path + 'cnn_{}_{}.txt'.format(target_layer, target_split)
    savetxt_compact(fout_name, new_cnn)


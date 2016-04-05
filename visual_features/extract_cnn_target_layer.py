"""
For each split
"""


import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys


n_regions = 4
n_region_per_folder = 1000 / n_regions # 250 for 4 regions per image.
total_n_imgs = 53689  # number of images
target_layer = 'fc7'
target_split = 'train'
cnn_dim = 4096
if target_layer == 'conv5':
    cnn_dim = 255 * 13 * 13

#
sys.path.append('/export/home2/NoCsBack/hci/susana/current/DeepFashion/projects/dress_project/')
from utils_local.utils_local import savetxt_compact

from caffe_setup import net, caffe
caffe.set_mode_gpu()

n_imgs_val = 4000
n_imgs_test = 1000
n_imgs_train = total_n_imgs - n_imgs_test - n_imgs_val

if target_split == 'train':
    n_rows = n_imgs_train * (n_regions + 1)
    cnn = np.zeros((n_rows, cnn_dim))
elif target_split == 'val':
    n_rows = n_imgs_val * (n_regions + 1)
    cnn = np.zeros((n_rows, cnn_dim))
elif target_split == 'test':
    n_rows = n_imgs_test * (n_regions + 1)
    cnn = np.zeros((n_rows, cnn_dim))
else:
    raise ValueError("must be either train, val, test")


# load json file
print "loading json file"
fname = '../../data/fashion53k/json/dataset_dress_all_{}.json'.format(target_split)
with open(fname, 'r') as f:
    data = json.load(f)
#
#
# # Set path for regions
img_path = '../../../IMAGES_plus_TEXT/DATASETS/'
region_img_path = '../../data/fashion53k/img_regions/{}_regions_imgs/'.format(n_regions)

# Set the filename to save the cnn features
cnn_out_path = '../../data/fashion53k/img_regions/{}_regions_cnn/per_split/'.format(n_regions)
fout_name = cnn_out_path + 'cnn_{}_{}.txt'.format(target_layer, target_split)
#

# cnn = np.zeros((200 * (n_regions + 1), cnn_dim))
i = 0
for item in data['items']:
    img_id = item['imgid']
    split = item['split']
    asin = item['asin']
    folder = item['folder']
    img_filename = item['img_filename']

    fname = img_path + folder + img_filename

    scores = net.predict([caffe.io.load_image(fname)])  # (1, 1000) 1000 classes from imagenet

    feat = net.blobs[target_layer].data[0]  #
    # cnn_dim = feat.flatten().shape[0]
    # print "cnn_dim", cnn_dim
    cnn[i, :] = feat.flatten()
    i += 1

    if i % 1 == 0:
        print i

    # Get the regions
    for region_id in range(n_regions):
        folder_id = img_id / n_region_per_folder
        region_fname = region_img_path + str(folder_id) + '/{}_{}.jpg'.format(img_id, region_id)
        scores = net.predict([caffe.io.load_image(region_fname)])  # (1, 1000) 1000 classes from imagenet

        feat = net.blobs[target_layer].data[0]
        cnn[i, :] = feat.flatten()
        i += 1

print "saving cnn features"
savetxt_compact(fout_name, cnn)
#

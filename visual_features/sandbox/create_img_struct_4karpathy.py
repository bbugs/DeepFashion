"""
I tried using this script to create the img struct for karpahty
However, the train file was too for the scipy.io to save.
I ended up writing the script in matlab, which worked a lot better.

Loading the json file in matlab takes way too long.  Perhaps I need another library. So I had to generate the cnn features separately for each split and make a csv file indicating which images belong to which split.

"""
import json
import numpy as np
import scipy.io as sio

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

data_2_matlab_train = []
data_2_matlab_val = []
data_2_matlab_test = []
index0 = 0
bias = np.ones((n_regions + 1, 1))
for item in data['items']:
    img_id = item['imgid']
    split = item['split']
    asin = item['asin']
    folder = item['folder']
    img_filename = item['img_filename']
    # print folder, img_filename
    # if img_id % 1000 == 0:
    #     print img_id

    index1 = index0 + (n_regions + 1)
    print img_id
    # print index0, index1

    cnn_slice = cnn[index0:index1, :]
    print cnn_slice.shape
    # print bias.shape
    cnn_2_store = np.append(cnn_slice, bias, 1)
    # print cnn_slice[:, 0:2]

    d = {}
    d['codes'] = cnn_2_store  # <n_regions x (cnn_dim + 1)>
    d['img_filename'] = img_filename
    d['img_id'] = img_id
    d['split'] = split
    d['asin'] = asin
    d['folder'] = folder

    if split == 'train':
        data_2_matlab_train.append(d)
    elif split == 'val':
        data_2_matlab_val.append(d)
    elif split == 'test':
        data_2_matlab_test.append(d)
    else:
        raise ValueError("must be train, val or test")

    index0 += (n_regions + 1)


print "saving struct file for matlab"
out_fname = rpath + 'matlab_structs/imgs/split_train_img.mat'
sio.savemat(out_fname, {'Img': data_2_matlab_train})

out_fname = rpath + 'matlab_structs/imgs/split_val_img.mat'
sio.savemat(out_fname, {'Img': data_2_matlab_val})

out_fname = rpath + 'matlab_structs/imgs/split_test_img.mat'
sio.savemat(out_fname, {'Img': data_2_matlab_test})

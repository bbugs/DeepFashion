"""
Read in cnn

Read in json file
"""
import json
import numpy as np
import scipy.io as sio

print "loading json file"
rpath = '../../data/fashion53k/'
fname = rpath + 'json/dataset_dress_all.json'
with open(fname, 'r') as f:
    data = json.load(f)


fname = rpath + 'img_regions/4_regions_cnn/cnn_fc7_medium.txt'
f = open(fname, "rb")
cnn = np.loadtxt(f, delimiter=",")
f.close()

print type(cnn)
print cnn.shape


target_split = 'test'
n_regions = 4
cnn_dim = 4096

data_2_matlab = []
index0 = 0
bias = np.ones((n_regions+1, 1))
for item in data['items'][0:40]:
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

    data_2_matlab.append(d)
    index0 += (n_regions + 1)


print "saving struct file for matlab"
out_fname = rpath + 'matlab_structs/imgs/split_{}_img.mat'.format(target_split)
sio.savemat(out_fname, {'Img': data_2_matlab})


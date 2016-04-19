"""
Compute cnn features for a set of images in img_paths

input: 
fname = dataset/paths_dress. txt (contains the paths of images)
layer = 'fc7' or 'conv5'


output
fout_name = 'cnn_dress.txt'  (contains cnn features in a matrix 4096 x len(img_paths)) 

"""
# Set the working directory
import os
os.chdir('/export/home2/NoCsBack/hci/susana/IMAGES_plus_TEXT/projects/')


import sys
sys.path.append('/export/home2/NoCsBack/hci/susana/IMAGES_plus_TEXT/projects/dress_project/')
from utils_local.utils_local import savetxt_compact

print sys.argv
# Set up the cuda library
# import os
# cmd = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64'
# os.system(cmd)

# Set up the layer I want to extract
target_layer = sys.argv[1]
print target_layer
# raw_input("Press a key to continue")

import pickle
import numpy as np
import matplotlib.pyplot as plt
# get the image paths
img_root = '../DATASETS/'
split = 'train_val'
fname = img_root + '/dress_attributes/data/paths/paths_dress_{}.txt'.format(split)
with open(fname, 'r') as f:
    paths = f.readlines()
img_paths = [img_root + img_path.strip('\n') for img_path in paths]
# img_paths = img_paths[0:2]

# Set name of out file
fout_name = '../DATASETS/dress_attributes/cnn/cnn_dress_' + target_layer + '_{}_transpose.txt'.format(split)


# Make sure that caffe is on the python path:
caffe_root = '/export/home2/NoCsBack/hci/susana/packages/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/export/home2/NoCsBack/hci/susana/packages/caffe/python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Run ./scripts/download_model_binary.py models/bvlc_reference_caffenet
# to get the pretrained CaffeNet model,

# load the net,
# specify test phase and CPU mode,
# and configure input preprocessing.

caffe.set_phase_test()
caffe.set_mode_gpu()
MODEL_FILE = caffe_root + '/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'  # binary file of pretrained model


net = caffe.Classifier(MODEL_FILE, PRETRAINED)

net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean
net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB



scores = net.predict([caffe.io.load_image(img_paths[0])])  # (1, 1000) 1000 classes from imagenet
feat = net.blobs[target_layer].data[0]  # (256 x 13 x13)
cnn_dim = feat.flatten().shape[0]
print "cnn_dim",  cnn_dim


print "len of img_paths", len(img_paths)
raw_input("Press any key to continue")

cnn = np.zeros((len(img_paths), cnn_dim))
i = 0
for img_path in img_paths:
    print i
    # caffe.io.load_image(img_paths[0])
    scores = net.predict([caffe.io.load_image(img_path)])  # (1, 1000) 1000 classes from imagenet

    # print dir(net)
    # print net.image_dims

    # The second fully connected layer, fc7 (rectified)
    feat = net.blobs[target_layer].data[0]  # (256, 13, 13)
    # print feat.flatten()
    # print feat.flatten().shape
    cnn[i,:] = feat.flatten()
    # raw_input('Press any key to continue')
    i += 1
    # fname = asin + '.pkl'
    # with open(fname, 'wb') as handle:
    #     pickle.dump(feat, handle)
    # np.savetxt(fname, feat, delimiter=',')
    # print feat
    # print feat.shape # (4096, 1, 1)
    # print feat[0:10]

# np.savetxt(open(fout_name,'w'), cnn, delimiter=',')

print "saving cnn features"
savetxt_compact(fout_name, cnn)


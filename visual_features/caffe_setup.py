import sys
import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/export/home2/NoCsBack/hci/susana/current/packages/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
sys.path.append('/export/home2/NoCsBack/hci/susana/current/packages/caffe/python')
import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_phase_test()
caffe.set_mode_gpu()
MODEL_FILE = caffe_root + '/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'  # binary file of pretrained model

net = caffe.Classifier(MODEL_FILE, PRETRAINED)

# print net.mean

net.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))  # ImageNet mean

# print net.mean

net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# print dir(caffe)
# print dir(net)

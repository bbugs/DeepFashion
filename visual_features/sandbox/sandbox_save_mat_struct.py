import numpy as np
import scipy.io as sio

dt = [('field1', 'f8'), ('field2', 'S10')]
arr = np.zeros((2,), dtype=dt)
print arr
# [(0.0, '') (0.0, '')]
# np.array([(0.0, ''), (0.0, '')],
#       dtype=[('field1', '<f8'), ('field2', 'S10')])

# arr[0]['field1'] = 0.5
# arr[0]['field2'] = 'python'
# arr[1]['field1'] = 99
# arr[1]['field2'] = 'not perl'
# sio.savemat('np_struct_arr.mat', {'arr': arr})
#
#
# arr = np.zeros((2,), dtype=dt)
#
# sio.savemat('out_name', {'Img': arr})

d0 = {}
d0['codes'] = np.zeros((5, 4))  #<n_regions x (cnn_dim + 1)>
d0['img_filename'] = 'filename'
d0['img_id'] = 4
d0['split'] = 'train'
d0['asin'] = 'asin'
d0['region_id'] = 2
d0['folder'] = 'folder'

d1={}
d1['codes'] = np.zeros((5, 4))  #<n_regions x (cnn_dim + 1)>
d1['img_filename'] = 'filename'
d1['img_id'] = 4
d1['split'] = 'train'
d1['asin'] = 'asin'
d1['region_id'] = 2
d1['folder'] = 'folder'

d = [d0, d1]

sio.savemat('test.mat', {'d': d})

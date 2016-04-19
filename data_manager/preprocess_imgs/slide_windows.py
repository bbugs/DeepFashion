import json
import os
from PIL import Image

n_regions = 4
n_region_per_folder = 1000 / n_regions  # 250 for 4 regions per image.  1000 is number of files per folder

# out_folder = '<n_regions>_regions/<folder_id>'  e.g. 4_regions/0
# folder_id = img_id / (n_img_per_folder)

# load json file
print "loading json file"
fname = '../../data/fashion53k/json/dataset_dress_all.json'
with open(fname, 'r') as f:
    data = json.load(f)


# iterate over images
rpath = '../../../IMAGES_plus_TEXT/DATASETS/'
out_path = '../../data/fashion53k/img_regions/{}_regions_imgs/'.format(n_regions)
for item in data['items']:
    img_id = item['imgid']
    split = item['split']
    asin = item['asin']
    folder = item['folder']
    img_filename = item['img_filename']
    # print folder, img_filename
    if img_id % 1000 == 0:
        print img_id
    # Open image
    fname = rpath + folder + img_filename
    pil_im = Image.open(fname)
    # Crop image
    w, h = pil_im.size
    y0 = 0
    y1 = h / n_regions
    # Specify folder to save the img regions
    folder_id = img_id / n_region_per_folder  # img_id / 250 when n_regions=4
    region_path = out_path + str(folder_id) + '/'
    if not os.path.exists(region_path):
        os.makedirs(region_path)
    # Get regions
    for region_id in range(n_regions):
        box = (0, y0, w, y1)
        # The region is defined by a tuple (left, upper, right, lower).
        # PIL uses (0, 0) in the upper left corner.
        y0 += h / n_regions
        y1 += h / n_regions
        region = pil_im.crop(box)

        region_fname = region_path + '{}_{}.jpg'.format(img_id, region_id)
        region.save(region_fname)



# for each image divide into regions by the height/10



# Crop the image into the regions



# save each image as img_id_region_id.jpg


# Every thousand regions, save in a new folder

rpath = '/export/home2/NoCsBack/hci/susana/current/IMAGES_plus_TEXT/DATASETS/dress_attributes/'

fname = rpath + '/data/images/BridesmaidDresses/B0009PDO0Y.jpg'

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch


def main():

    # loading lena image
    img = skimage.data.lena()

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    print dir(regions)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # save all candidates

    i = 0
    for c in candidates:
        print i
        x, y, w, h = c  # x, y, width, height

        print x, y, w, h
        cropped = img[x:x+w, y:y+h]
        fname = 'regions/region_{}.jpg'.format(i)
        i += 1
        skimage.io.imsave(fname, cropped)


    # draw rectangles on the original image
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(img)
    # for x, y, w, h in candidates:
    #     print x, y, w, h
    #     rect = mpatches.Rectangle(
    #         (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    #     ax.add_patch(rect)
    #
    # plt.show()

if __name__ == "__main__":
    main()

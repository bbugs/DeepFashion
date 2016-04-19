

from PIL import Image
import selectivesearch
import skimage

fname = 'B006Y2L2CI.jpg'
# pil_im = Image.open(fname)
#
# print pil_im.shape

dress_img = skimage.io.imread(fname)

print "here this is "
print dress_img.shape

# box = (100, 100, 400, 400)
# region = pil_im.crop(box)
#
# region.save('region.jpg')


# perform selective search
img_lbl, regions = selectivesearch.selective_search(
    dress_img, scale=500, sigma=0.9, min_size=10)



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

    print x, y, w, h  # (left, top, right, bottom)
    cropped = dress_img[y: y + h, x: x + w]
    print cropped.shape
    fname = 'regions/region_{}.jpg'.format(i)

    skimage.io.imsave(fname, cropped)

    i += 1

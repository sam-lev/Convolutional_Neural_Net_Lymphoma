import datapack

ds = datapack.lymphoma2('train', dir='../data')

ds.reset_state()

c = 0
im = []
for d in ds:
    if c == 0:
        im = d
    c+=1

im_aug = datapack.medical_aug.hematoxylin_eosin_aug(low = 0.6, high = 1.4).apply_image(im[0])

from PIL import Image

aug = Image.fromarray(im_aug)
aug.show()

img_og = Image.fromarray(im[0])
img_og.show()

import datapack
import tensorflow as tf
from tensorpack import *
import numpy as np

ds0 = datapack.lymphoma2('train', dir='../data')
ds = datapack.lymphoma2('train', dir='../data')

batch_size = ds.size()
print("total im: ", batch_size)


ds_unAug = BatchData(ds0, 1, remainder=True)

c = 0
im = []
im_aug = []
label = 2
for dp_ua in ds_unAug:
    print("Data shape OG: ", len(dp_ua), ' elm 0 size ', dp_ua[0].shape)
    if c == 0:
        im = dp_ua[0][0,:,:,:]
    c+=1
    
augmentors = [
    datapack.HematoEAug((3.1, 3.2, np.random.randint(2**32-1))),
    #datapack.HematoEAug((0.1, 0.2, np.random.randint(2**32-1))),
    datapack.NormStainAug(),]

augmentor = imgaug.AugmentorList(augmentors)
ds = MultiThreadMapData(ds,
                        nr_thread=2,
                        map_func=lambda dp: [augmentor.augment(dp[0]),
                                                    dp[1]],
                        buffer_size=100)

#ds = AugmentImageComponent(ds, augmentors)
ds = PrefetchDataZMQ(ds, nr_proc=1)
ds = BatchData(ds, 1, remainder=True)

ds.reset_state()

c = 0
im_aug = []
for dp in ds:
    print("Data shape: ", len(dp), ' elm 0 size ', dp[0].shape)
    if c == 0:
        im_aug = dp[0][0,:,:,:]
        label = dp[1]
    c+=1

im = im.astype('uint8')
im_aug = im_aug.astype('uint8')

#im_aug_he = augmentors[0]._augment(im)
#im_aug_he_2 = augmentors[1]._augment(im)

#datapack.medical_aug.hematoxylin_eosin_aug(low = 0.6, high = 1.4).apply_image(im)
#im_aug_norm = datapack.medical_aug.normalize_staining().apply_image(im_aug_he)

from PIL import Image

print(" Class of Image: ", label)

img_aug = Image.fromarray(im_aug)
img_aug.show()
#aug_he = Image.fromarray(im_aug_he)
#aug_he.show()
#aug_he_2 = Image.fromarray(im_aug_he_2)
#aug_he_2.show()
#aug_norm = Image.fromarray(im_aug_norm)
#aug_norm.show()

img_og = Image.fromarray(im)
img_og.show()

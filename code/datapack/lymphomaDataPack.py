#!/home/sci/samlev/bin/bin/python3

#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t
#SBATCH --mem=30G
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)        
#SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:1

import numpy as np
import tensorflow as tf
import argparse
import os

import six
from six.moves import range
from abc import abstractmethod, ABCMeta
import threading

#from scipy.misc import imsave
import pickle as pickle

import random as rnd
from datapack import quality_random_crop

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
#from tensorpack.utils.utils import get_rng

from tensorpack.dataflow.base import RNGDataFlow
#from lymphomaflow import DataFlow
#from lymphomaflow import RNGDataFlow
#RNGDataFlow = RNGDataFlow(DataFlow)
from PIL import Image
import cv2

#sys.path.append(os.getcwd())
prediction_dir = "/home/sci/samlev/convNet_Lymphoma_Classifier/data/Unknowns/predictions/"

"""
Data flow similar to cifar10 http://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/dataflow/dataset/cifar.html 
structure explained here: https://www.cs.toronto.edu/~kriz/cifar.html
Summary: Data stream capable of parallel feed with images 32x32x32x3 (RGB) 

 data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
 labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

Example Call: python denseNet_Lymphoma.py --tot train --drop_1 2 --drop_2 4 --depth 26 --max_epoch 1

"""
val_or_test = [['blackbox', False], ['test', True], ['dlbcl_test', False]][1]

def save_original(img, save_dir = None, name="temp"):
    img = np.transpose(img, [1, 2, 0])
    im = Image.fromarray(np.asarray(img).astype('uint8'),'RGB')
    im.save(prediction_dir+save_dir+'_Predictions/'+save_dir+'_Originals/'+str(name)+'.jpeg')
    im.close()
    
def read_lymphoma(filenames,  original_dir=None):
    num_show=5
    ret = []
    for fname in filenames:

        #fo = open(fname, 'rb')

        #if six.PY3:
        with open(fname, 'rb') as fo:
            try:
                dic = pickle.load(fo)
            except UnicodeDecodeError:  #python 3.x
                fo.seek(0)
                dic = pickle.load(fo, encoding='latin1')
                
        #dic = pickle.load(fo,encoding='latin1')

        #else:
        #dic = pickle.load(fo, encoding='bytes')
        data = dic['data']
        label = dic['labels']
        fo.close()
        for k in range(len(data)):
            # resize to divisable format for convolutions
            scaleSize = 224,224
            imSize = 672
            #randPos = rnd.choice([0, 50, 100, 200, 300])
            #img = data[k][:, randPos:(randPos+imSize), randPos:(randPos+imSize)] #:32, :32] #size currently (927,1276,3)

            if original_dir:
                save_original(data[k][:,:, :], save_dir=original_dir, name=k)
            quality_cropper = quality_random_crop(data[k][:,:, :], imSize)        
            img = quality_cropper.random_crop_select()  
            
            # make rgb feasible
            img = np.transpose(img, [1, 2, 0])
            img_ref = img
            # format required for DataFlow
            img = Image.fromarray(img,'RGB')
            img = img.resize(scaleSize, Image.ANTIALIAS)#thumbnail(scaleSize) #image.resize(THUMB_SIZE, Image.ANTIALIAS)
            if num_show < 5:
                img.show()
                num_show=5
            img = np.asarray(img).reshape((224,224,3))

            #img = np.transpose(img, [1,2,0])
            #test = Image.fromarray(img, 'RGB')
            
            #img.tofile(str(k)+'.raw')
            #test.show()
            #img = np.transpose(img, [1,2,0])
            ret.append([img, label[k]])

    return ret

def get_filenames(dir, unknown_dir = None):
    
    filenames = [os.path.join(dir, 'train', 'batch_data_%d.p' % i) for i in range(6)] #as more data is obtained
    if val_or_test[0] == 'test':
        filenames.append([os.path.join(dir, val_or_test[0],'test_batch_%d.p' % i) for i in range(1)]) #'blackbox' #  'test_batch_%d.p'
    else:
        if unknown_dir:
            filenames.append([os.path.join(dir, val_or_test[0],unknown_dir+'.p')])
        else:
            group = ['A_0','B_0','C_0','D_0','E_0','F_0','G_0','H_0','I_0','J_0','J_0_0','J_1_0','K_0','L_0','M_0','N_0','O_0','P_0','Q_0','R_0','S_0']
            #filenames.append([os.path.join(dir, val_or_test[0],'G_%d.p' % i) for i in range(1)])
            filenames.append([os.path.join(dir, val_or_test[0],group[i]+'.p') for i in [12]])#range(1)])
    return filenames

class lymphomaBase( RNGDataFlow ):
    # Data flow for lymphoma images.
    # yields [ Image, Label ]
    # image: 900x900x3 in range [0,255]
    # label: int either 0 or 1
    def __init__(self, train_or_test, shuffle=True, dir=None, lymphoma_num_classes=2,unknown_dir = None, original_dir=None):
        assert train_or_test in ['train', 'test']
        assert lymphoma_num_classes == 2 or lymphoma_num_classes == 10
        self.lymphoma_num_classes = lymphoma_num_classes


        if dir is None:
            dir = '../data' #and changes in behavor for more classes here
        fnames = get_filenames(dir, unknown_dir=unknown_dir)
        
        if train_or_test == 'train':
            self.fs = fnames[:-1]
        else:
            self.fs = [f for f in fnames[-1]]
        for f in self.fs:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
        self.train_or_test = train_or_test
        self.data = read_lymphoma(self.fs,original_dir=original_dir) #different classes changes here ect..
        self.dir = dir
        self.shuffle = shuffle
        
    def size(self):
        return len(self.data)
    
    # Required get_data for DataFlow
    def get_data(self):
        image_data = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(image_data)
        for i in image_data:
            yield self.data[i]

    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 927x1276
        """
        fnames = get_filenames(self.dir)
        all_imgs = [x[0] for x in read_lymphoma(self.fs)]
        arr = np.array(all_imgs, dtype='float32')
        mean = np.mean(arr, axis=0)
        return mean
    def get_per_channel_mean(self):
        """
        return three values as mean of each channel
        """
        mean = self.get_per_pixel_mean()
        return np.mean(mean, axis=(0, 1))
            
class lymphoma2(lymphomaBase):
    """
    Produces [image, label] in Cifar10 dataset,
    image is 900x900x3 in the range [0,255].
    label is an int.
    """
    def __init__(self, train_or_test, shuffle= val_or_test[1], dir=None, unknown_dir=None,original_dir=None):

        """
        Args:
            train_or_test (str): either 'train' or 'test'.
            shuffle (bool): shuffle the dataset.
        """
        super(lymphoma2, self).__init__(train_or_test, shuffle, dir, 2,unknown_dir, original_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Takes directory of training images or testing images')
    parser.add_argument('-traindir', type=str, nargs = 1, help ='path to training data', required = False)
    parser.add_argument('-testdir', type=str, nargs = 1, help ='path to testing data ', required = False)
    parser.add_argument('-tot', type=str, nargs = 1, help ="Either 'train' or 'test' ", required = False)
    args = parser.parse_args()
    
    if args.traindir[0]:
        dir = args.traindir[0]
    if args.testdir[0]:
        dir = args.testdir[0]
    print(dir)

    ds = lymphoma2('train', dir = dir)
    mean = ds.get_per_channel_mean()
    print(mean)

    import cv2
    ds.reset_state()
    for i, dp in enumerate(ds.get_data()):
        if i == 100:
            break
        img = dp[0]
        cv2.imwrite("{:04d}.jpg".format(i), img)
    # get pixel mean here?
    print('here')

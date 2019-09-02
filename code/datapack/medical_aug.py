#!/usr/bin/env python3
"""Extract deep CNN features from a set of images and dump them as Numpy arrays image_file_name.npy"""

from tensorpack.dataflow import imgaug

import argparse
import numpy as np
import cv2
from scipy import ndimage
from os.path import basename, join, exists
from os import makedirs
#from threaded_generator import threaded_generator
from time import time
import sys
np.random.seed(101)

PATCH_SIZES = [224, 224] #[672,672]
SCALES = [0.5]

DEFAULT_INPUT_DIR = "data/train"
DEFAULT_PREPROCESSED_ROOT = "data/preprocessed/train"

PATCHES_PER_IMAGE = 20
AUGMENTATIONS_PER_IMAGE = 50
COLOR_LO = 0.7
COLOR_HI = 1.3
BATCH_SIZE = 16     # decrease if necessary

NUM_CACHED = 160

class NormStainAug(imgaug.ImageAugmentor):
    def __init__(self):
        super(NormStainAug, self).__init__()
        self._init(locals())
        
    
    def get_transform(self, img):
        return normalize_staining()
    
    def apply_coords(self, coords):
        return coords
    
    def _get_augment_params(self, img):
        return np.random.randint(100)
    
    def _augment(self, img, _):
        t = self.get_transform(img).apply_image(img)
        return t
    
    #def reset_state(self):
    #    super(NormStainAug, self).reset_state()
            
class ZoomAug(imgaug.ImageAugmentor):
    def __init__(self, param = (10, None)):
        super(ZoomAug, self).__init__()
        self.zoom = zoom[0]
        self.seed = seed[1]
        self._init(locals())
        
    def get_transform(self, img):
        return zoom_transform(self.zoom, self.seed)
    
    def	apply_coords(self, coords):
        return coords
    
    def _get_augment_params(self, img):
        return (self.zoom, self.seed)
    
    def	_augment(self, img, param = (10, None)):
        self.zoom = param[0]
        self.seed = param[1]
        t = self.get_transform(img).apply_image(img)
        return t

class HematoEAug(imgaug.ImageAugmentor):
    def __init__(self, param = (0.7, 1.3, None)):
        super(HematoEAug, self).__init__()
        self.low = param[0]
        self.high = param[1]
        self.seed = param[2]
        self._init(locals())
        
        
    def get_transform(self, img):
        return hematoxylin_eosin_aug(self.low, self.high, self.seed)
    
    def apply_coords(self, coords):
        return coords
    
    def _get_augment_params(self, img):
        return (self.low, self.high, self.seed)
    
    def	_augment(self, img, param = (0.7, 1.3, None)):
        self.low = param[0]
        self.high = param[1]
        self.seed = param[2]
        t = self.get_transform(img).apply_image(img)
        return t

class normalize_staining(imgaug.transform.ImageTransform):
    def __init__(self):
        super(normalize_staining, self).__init__()
        self._init(locals())
        
    def apply_image(self, img):
        
        """
        Adopted from "Classification of breast cancer histology images using Convolutional Neural Networks",
        Teresa Araújo , Guilherme Aresta, Eduardo Castro, José Rouco, Paulo Aguiar, Catarina Eloy, António Polónia,
        Aurélio Campilho. https://doi.org/10.1371/journal.pone.0177544
        Performs staining normalization.
        # Arguments
        img: Numpy image array.
        # Returns
        Normalized Numpy image array.
        """
        Io = 240
        beta = 0.15
        alpha = 1
        HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        maxCRef = np.array([1.9705, 1.0308])
        
        h, w, c = img.shape
        img = img.reshape(h * w, c)
        OD = -np.log((img.astype("uint16") + 1) / Io) #img.astype("uint16")
        ODhat = OD[(OD >= beta).all(axis=1)]
        W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))
        
        Vec = -V.T[:2][::-1].T  # desnecessario o sinal negativo
        That = np.dot(ODhat, Vec)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax])
        else:
            HE = np.array([vMax, vMin])
            
        HE = HE.T
        Y = OD.reshape(h * w, c).T
        
        C = np.linalg.lstsq(HE, Y, rcond=None)
        maxC = np.percentile(C[0], 99, axis=1)
        
        C = C[0] / maxC[:, None]
        C = C * maxCRef[:, None]
        Inorm = Io * np.exp(-np.dot(HERef, C))
        Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype("uint8")
        
        return Inorm
    
    def apply_coords(self, coords):
        return coords
    
class hematoxylin_eosin_aug(imgaug.transform.ImageTransform):
    def __init__(self, low=0.7, high=1.3, seed=None):
        super(hematoxylin_eosin_aug, self).__init__()
        self.low = low
        self.high = high
        self.seed = seed
        self._init(locals())
        
    def apply_image(self, img):
        low = self.low
        high = self.high
        seed = self.seed
        """
        "Quantification of histochemical staining by color deconvolution"
        Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.
        http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf
        Performs random hematoxylin-eosin augmentation
        # Arguments
        img: Numpy image array.
        low: Low boundary for augmentation multiplier
        high: High boundary for augmentation multiplier
        # Returns
        Augmented Numpy image array.
        """
        D = np.array([[1.88, -0.07, -0.60],
                      [-1.02, 1.13, -0.48],
                      [-0.55, -0.13, 1.57]])
        M = np.array([[0.65, 0.70, 0.29],
                      [0.07, 0.99, 0.11],
                      [0.27, 0.57, 0.78]])
        Io = 240
        
        h, w, c = img.shape
        OD = -np.log10((img.astype("uint16") + 1.) / Io)#.astype("uint16")
        C = np.dot(D, OD.reshape(h * w, c).T).T
        r = np.ones(3)
        r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)
        img_aug = np.dot(C, M) * r
        
        img_aug = Io * np.exp(-img_aug * np.log(10)) - 1
        img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype("uint8")
        
        return img_aug
    
    def apply_coords(self, coords):
        return coords

class zoom_transform(imgaug.transform.ImageTransform):
    def __init__(self, zoom, seed = None):
        super(zoom_transform, self).__init__()
        self.zoom = zoom
        self.seed = seed
        self._init(locals())
        
    def apply_image(self, img):
        zoom = self.zoom
        seed = self.seed
        """Performs a random spatial zoom of a Numpy image array.
        # Arguments
        img: Numpy image array.
        zoom_var: zoom range multiplier for width and height.
        seed: Random seed.
        # Returns
        Zoomed Numpy image array.
        """
        scale = np.random.RandomState(seed).uniform(low=1 / zoom_var, high=zoom_var)
        resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return resized_img
    
    def apply_coords(self, coords):
        return coords

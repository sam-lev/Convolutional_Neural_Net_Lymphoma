#!/home/sci/samlev/bin/bin/python3                                              

#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t                       
#SBATCH --mem=30G                                                             
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)                                                            
#SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)   
#SBATCH --gres=gpu:1

import numpy as np
import matplotlib.pyplot as plt
#from skimage import io
from PIL import Image

import random as rnd

class quality_random_crop:

    def __init__(self, image, image_size=672):
        self.entire_image = image
        self.image_size = image_size
        
    def show_image(self, image):
        plt.figure(num=None, dpi=200)
        plt.imshow(image, cmap=plt.cm.Greys_r)

    def get_tile_image(self, image):
        imSize=self.image_size
        img_tiles = []
        stride = 272
        if image.shape[1] > 1000 and image.shape[2] > 1500:
            rand_y = rnd.choice([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275])
            rand_pos = rnd.choice([150, 150+stride, 150+(2*stride), 150+(3*stride)])
        else:
            stride = 100
            rand_y = rnd.choice([0, 25, 50, 75, 100, 125])
            rand_pos = rnd.choice([150, 150+stride, 150+(2*stride), 150+(3*stride)])  
        img = image[:,(150+rand_y):((150+rand_y)+imSize), rand_pos:(rand_pos+imSize)] #:32, :32] #size currently (927,1276,3)
        img_tiles.append(img)
        return  img_tiles[0]


    
    
    def gaussian_fit(self, image, plot = False):
        n, bins_ = np.histogram(image.flatten())
        mids = 0.5*(bins_[1:] + bins_[:-1])
        mu = np.average(mids, weights=n)
        var = np.average((mids - mu)**2, weights=n)
        sigma = np.sqrt(var)
        right_inflection = mu+sigma
        return n, mu, sigma, var#, right_inflection)

    #fname = 'white_case'#'dl_white'
    #image =  io.imread(fname+'.jpg')

    def hist_show(self, image, fname=None):
        #image =  io.imread(fname+'.jpg')
        hist, mu, sigma, var = gaussian_fit(image)
        rng = np.random.RandomState(10) 
        a = np.hstack((rng.normal(size=1000)))
        plt.hist(a, bins='auto')
        plt.title(r' mu=%.3f, \sigma=%.3f$' %(mu, sigma))
        plt.show()

    def tile_show(self):
        image=self.entire_image
        i = 1
        for img in get_tile_image(image ):
            #img = np.transpose(img, [1, 2, 0])
            img = Image.fromarray(img,'RGB')
            img.save(str(i)+'.jpg')
            i+=1
            img.show()
        
    #tile_show()
    #hist_show('white')
    
    def content_check(self, image):
        hist, mu, sigma, var = gaussian_fit(image)

        if mu > 155:
            return 1, image
        else:
            return 0 , image

    def random_crop_select(self):
        full_image = self.entire_image
        good_crop = 1
        image = self.get_tile_image(full_image)
        iter_bound = 10
        i = 0
        while not good_crop and i < iter_bound:
            i += 1
            image = self.get_tile_image(full_image)
            good_crop, image = self.content_check(image)
        return image

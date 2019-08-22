import _pickle as pickle
from PIL import Image
import numpy as np
import argparse
import os

#Example: python storeData.py -dirbl data/BL/ -dirdl data/DlBCL/ -pout data/ -dirtest data/Unknowns/

#         python storeData.py -dirbl data/BL/ -dirdl data/DlBCL/ -pout data/ -dirtest_bl data/BL_sub/ -dirtest_dlbcl data/DLBCL_sub/
parser = argparse.ArgumentParser()
parser.add_argument('-dirbl', type = str, help='BL train directory')
parser.add_argument('-dirdl', type = str, help='DLBCL train directory')
parser.add_argument('-pout', type = str, help='Pickle output directory')
parser.add_argument('-dirtest', type = str, help='Directory of test data')
parser.add_argument('-dirtest_bl', type = str, help='Directory of test data')
parser.add_argument('-dirtest_dlbcl', type = str, help='Directory of test data')
args = parser.parse_args()

dir_bl = args.dirbl
dir_dlbcl = args.dirdl
dir_test = args.dirtest
dir_test_bl = args.dirtest_bl
dir_test_dlbcl = args.dirtest_dlbcl
pout = args.pout

"""
Stores data similar to Cifar10.
 data -- a numpy array of uint8s. Each row of the array stores full colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
 labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
"""
# first BL then DLBCL, i.e. [BL images, DLBCL images]
# labels 0 bl, labels 1 dlbcl
# labels all 1 for unknown

##
## Example Run: python storeData.py -dirbl ./data/BL -dirdl ./data/DLBCL -dirtest ./data/Unknowns -dirtest_bl ./data/BL_sub -dirtest_dlbcl ./data/DLBCL_sub -pout ./data
##
def get_filenames():

    #train set
    filenames_bl = [os.path.join(dir_bl, f ) for f in os.listdir(dir_bl) if f.endswith('.jpg')]
    labels_bl = [0 for i in range(len(filenames_bl))]
    filenames_dlbcl = [os.path.join(dir_dlbcl, f) for f in os.listdir(dir_dlbcl) if f.endswith('.jpg')]
    labels_dlbcl = [1 for i in range(len(filenames_dlbcl))]
    labels_train =  labels_dlbcl + labels_bl 
    filenames =    filenames_dlbcl +filenames_bl 

    #blackbox/sample test sets
    filenames_blackbox = [os.path.join(dir_test, f) for f in os.listdir(dir_test) if f.endswith('.jpg')]
    filenames_test_bl = [os.path.join(dir_test_bl, f) for f in os.listdir(dir_test_bl) if f.endswith('.jpg')]
    filenames_test_dlbcl = [os.path.join(dir_test_dlbcl, f) for f in os.listdir(dir_test_dlbcl) if f.endswith('.jpg')]
    # assign label 1, i.e. dlbcl, for all testing data
    labels_blackbox = [1 for i in range(len(filenames_blackbox))]
    
    #Validation set
    labels_test_bl = [0 for i in range(len(filenames_test_bl))]
    labels_test_dlbcl = [1 for i in range(len(filenames_test_dlbcl))]
    filenames_validation = filenames_test_bl + filenames_test_dlbcl
    labels_validation = labels_test_bl + labels_test_dlbcl
    
    return (filenames, labels_train, filenames_validation, labels_validation, filenames_blackbox, labels_blackbox)#filenames_test, labels_test)#

def pickle_data():
    fnames_train, labels_train, fnames_test, labels_test, fnames_blackbox, labels_blackbox = get_filenames()
    
    #train set
    im_sample = Image.open(fnames_train[0])
    im_sample = np.array(im_sample)
    im_train = np.array([[im_sample[:,:,0]] + [im_sample[:,:,1]] + [im_sample[:,:,2]]], np.uint8)
    for f in fnames_train[1:]:
        im = np.array(Image.open(f))
        r = im[:,:,0] #Slicing to get R data
        g = im[:,:,1] #Slicing to get G data 
        b = im[:,:,2] #Slicing to get B data
        formated_im = np.array([[r] + [g] + [b]], np.uint8)
        im_train = np.append(im_train, formated_im, 0)

    
    #validation set
    im_sample = Image.open(fnames_test[0])
    im_sample = np.array(im_sample)
    im_test = np.array([[im_sample[:,:,0]] + [im_sample[:,:,1]] + [im_sample[:,:,2]]], np.uint8)
    for f in fnames_test[1:]:
        im = np.array(Image.open(f))
        r = im[:,:,0] #Slicing to get R data
        g = im[:,:,1] #Slicing to get G data
        b = im[:,:,2] #Slicing to get B data
        formated_im = np.array([[r] + [g] + [b]], np.uint8)
        im_test = np.append(im_test, formated_im, 0)

    """
    #blackbox set
    im_sample = Image.open(fnames_blackbox[0])
    im_sample = np.array(im_sample)
    im_blackbox = np.array([[im_sample[:,:,0]] + [im_sample[:,:,1]] + [im_sample[:,:,2]]], np.uint8)
    for f in fnames_blackbox[1:]:
        im = np.array(Image.open(f))
        r = im[:,:,0] #Slicing to get R data
	g = im[:,:,1] #Slicing to get G data
        b = im[:,:,2] #Slicing to get B data
        formated_im = np.array([[r] + [g] + [b]], np.uint8)
        im_blackbox = np.append(im_blackbox, formated_im, 0)
    """
    
        
    training_data = {'data': im_train, 'labels' : labels_train}
    testing_data = {'data': im_test, 'labels' : labels_test}
    #blackbox_data = {'data': im_blackbox, 'labels': labels_blackbox}

    #training to file
    batch_data_pickle = os.path.join(pout, 'train','batch_data_0.p')
    with  open(batch_data_pickle, 'wb') as p:
        pickle.dump(training_data, p)


    
    #validation to file
    test_batch_pickle = os.path.join(pout,'test','test_batch_0.p')# 'blackbox', 'test_batch_0.p')#
    with  open(test_batch_pickle, "wb") as p:
        pickle.dump(testing_data, p)

    """
    #blackbox to file
    blackbox_batch_pickle = os.path.join(pout,'blackbox','blackbox_0.p')# 'blackbox', 'test_batch_0.p')#
    with  open(blackbox_batch_pickle, "wb") as p:
        pickle.dump(blackbox_data, p)
    """
    
    #checks
    """
    train_check = pickle.load( open(batch_data_pickle, "rb"))
    if ((np.array_equal(train_check[ 'data' ] ,im_train))
        and train_check['labels'] == labels_train):
        print('Training data pickled')
    else:
        print("Error pickeling training data")
        
    test_check = pickle.load( open(test_batch_pickle, "rb"))
    if (  (np.array_equal(test_check['data'], im_test))
          and test_check['labels'] == labels_test):
        print('Testing data pickled')
    else:
        print("Error pickeling testing data")

    blackbox_check = pickle.load( open(blackbox_batch_pickle, "rb"))
    if (  (np.array_equal(blackbox_check['data'], im_blackbox))
          and blackbox_check['labels'] == labels_blackbox):
        print('Black box data pickled')
    else:
        print("Error pickeling blackbox data")
    """
    return 0
    

pickle_data()

#!/home/sci/samlev/bin/bin/python3.5

#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t
#SBATCH --mem=120G
#SBATCH -o model_shallow.out-%j # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e model_shallow.err-%j # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:1

import numpy as np
import tensorflow as tf
import argparse
import os
import sys

import sklearn.metrics
import matplotlib.pyplot as plt
import io
import itertools

# Personal Data Flow
sys.path.append(os.getcwd())
import datapack 

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
#from tensorpack.tfutils.symbolic_functions import prediction_incorrect
from tensorpack.tfutils.sesscreate import NewSessionCreator

#from tensorpack.dataflow import LocallyShuffleData

#from tensorpack.dataflow import *
import multiprocessing as mp
import copy

from PIL import Image


""" slurm gpu test 
"""
def get_available_gpus():
   from tensorflow.python.client import device_lib
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type == 'GPU']



"""
Lymphoma Deep Convolutional Neural Net Based on http://arxiv.org/abs/1608.06993
Code similar to Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
See also: https://github.com/YixuanLi/densenet-tensorflow/blob/7793c03a9c9dc009e4151aa4b7a74f0e62583973/cifar10-densenet.py
Dataflow formated similar to cifar10 http://tensorpack.readthedocs.io/en/latest/_modules/tensorpack/dataflow/dataset/cifar.html 
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
 @article{Huang2016Densely,
 		author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
 		title = {Densely Connected Convolutional Networks},
 		journal = {arXiv preprint arXiv:1608.06993},
 		year = {2016}
 }
Call Example:  
>> Train listing GPU to use and basic training param:
python3 denseNet_Lymphoma.py --gpu 0,1 (--num_gpu 4) (--load model-xxx) --drop_1 100 --drop_2 200 --depth 40 --max_epoch 368 --tot train
>> Train Using Desired number of GPU and loading model partially trained:
python3 denseNet_Lymphoma.py --num_gpu 8 --tot train --model_name 8_gpu_d1_58_d2_98_phase_3 --lr_0 0.01 --drop_1 58 --lr_1 0.001 --drop_2 98 --lr_2 0.0001 --max_epoch 134 --gpu_frac 0.99 --batch_size 2 --load train_log/4_gpu_d1_60_d2_160_d3_220_max_256-first63-second113-max149/model-398208
>> Predict Class of files in folder '../data/Unknon/' + unknown_dir 
>> and write prediction to '../data/Unknown/'+unknown_dir+'/predictions/'
python3 denseNet_Lymphoma.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load train_log/8_gpu_d1_58_d2_98_phase_3-first58-second98-max134/model-523136 --unknown_dir A
"""

class Model(ModelDesc):
   def __init__(self, depth, image_size, lr_init, kernels, kernel_size, expansion, class_0, class_1, drop_rate, train_or_test):
      super(Model, self).__init__()
      self.step = tf.train.get_or_create_global_step()
      self.N = int((depth - 4)  / 3)
      self.image_size = image_size
      self.growthRate = expansion
      self.filters_init = kernels
      self.lr_init = lr_init
      self.drop_rate = drop_rate
      self.train_or_test= train_or_test
      if class_0 == class_1:
         self.class_0 = 1.0
         self.class_1 = 1.0
      else:
         self.class_0 = float(class_0)
         self.class_1 = float(class_1)
         
      self.kernel_size = kernel_size
      print(">>> class 0: ", self.class_0)
      print(">>> class 1: ", self.class_1)
      
   def _get_inputs(self):
      return [InputDesc(tf.float32, [None, self.image_size, self.image_size, 3], 'input'),
              InputDesc(tf.int32, [None], 'label')
      ]
   
   nondep = """def inputs(self):
   return [tf.TensorSpec((None, 224, 224, 3), tf.uint8, 'input'),tf.TensorSpec((None,), tf.int32, 'label')]"""
   
   def _build_graph(self, input_vars):
      image, label = input_vars
      image = tf.image.convert_image_dtype(image, dtype = tf.float32)
      
      tf.summary.image("Input Image", image[0:2], max_outputs=2)#.astype("uint8"))
      def conv(name, l, channel, stride):
         #rand_seed = np.random.randint(2**32-1)
         #np.random.seed(None)
         conv2d_xav = Conv2D(name, l, channel, self.kernel_size, stride=stride,
                             nl=tf.identity, use_bias=False,
                             W_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=False))
         #np.random.seed(rand_seed)
         return conv2d_xav
      
      def add_layer(name, l):
         shape = l.get_shape().as_list()
         in_channel = shape[3]
         with tf.variable_scope(name) as scope:
            c = BatchNorm('bn1', l)
            c = tf.nn.relu(c)
            c = conv('conv1', c, self.growthRate, 1)
            l = tf.concat([c, l], 3)
         return l
      
      def add_transition(name, l, drop_rate, training):
         shape = l.get_shape().as_list()
         in_channel = shape[3]
         with tf.variable_scope(name) as scope:
            l = BatchNorm('bn1', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
            l = AvgPooling('pool', l, 2)
            l = tf.layers.dropout(l, rate = drop_rate, training=training)
         return l
      
      def dense_net(name):
         l = conv('conv0',image,  self.filters_init , 1)
         
         #tf.summary.image("Convolutional Image", l[0,:,:,0:1])#[0,:,:,:])
         with tf.variable_scope('block1') as scope:
            
            for i in range(self.N):
               l = add_layer('dense_layer.{}'.format(i), l)
            l = add_transition('transition1', l ,self.drop_rate, self.train_or_test)
            
         with tf.variable_scope('block2') as scope:
            for i in range(self.N):
               l = add_layer('dense_layer.{}'.format(i), l)
            l = add_transition('transition2', l, self.drop_rate, self.train_or_test)
         
         with tf.variable_scope('block3') as scope:
            
            for i in range(self.N):
               l = add_layer('dense_layer.{}'.format(i), l)
         l = BatchNorm('bnlast', l)
         l = tf.nn.relu(l)
         l = GlobalAvgPooling('gap', l)
         logits = FullyConnected('linear', l, out_dim=2, nl=tf.identity)
         
         return logits
      
      non_deptrincated="""def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
      with tf.name_scope('prediction_incorrect'):
      x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
      return tf.cast(x, tf.float32, name=name)"""
      
      logits = dense_net("dense_net") #map probabilities to real domain
      
      prob = tf.nn.softmax(logits, name='output')  #a generalization of the logistic function that "squashes" a K-dim vector z  of arbitrary real values to a K-dim vector sigma( z ) of real values in the range [0, 1] that add up to 1.
      factorbl = (self.class_0+self.class_1)/(2.0*self.class_0)
      factordl = (self.class_0+self.class_1)/(2.0*self.class_1)
      class_weights = tf.constant([factorbl, factordl])#factor,(1-factor)])#factor, 1.0-factor]) #dl 730 bl 1576
      weights = tf.gather(class_weights, label)
      
      cost = tf.losses.sparse_softmax_cross_entropy(label, logits, weights=weights) #False positive 3* False negatives so adjust weight by factor
      cost = tf.reduce_mean(cost, name='cross_entropy_loss') #normalize
      
      wrong = prediction_incorrect(logits, label)

      # monitor training error
      add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
      
      # weight decay on all W
      wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
      add_moving_summary(cost, wd_cost)
      
      add_param_summary(('.*/W', ['histogram']))   # monitor W
      
      self.cost = tf.add_n([cost, wd_cost], name='cost')
   
   
   non_depricated = """def build_graph(self, image, label):
   image = image
   label = label
   #image = image / 128.0 - 1
   def conv(name, l, channel, stride):
   rand_seed = np.random.randint(2**32-1)
   np.random.seed(101)
   conv2d_xav = Conv2D(name, l, channel, 6, stride=stride,
   nl=tf.identity, use_bias=False,
   W_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_AVG', uniform=False)) #factor=np.sqrt(2.0/(6+channel)), mode='FAN_IN', uniform=False, seed=None, dty
   np.random.seed(rand_seed)
   return conv2d_xav
   
   def add_layer(name, l):
   shape = l.get_shape().as_list()
   in_channel = shape[3]
   with tf.variable_scope(name) as scope:
   c = BatchNorm('bn1', l)
   c = tf.nn.relu(c)
   c = conv('conv1', c, self.growthRate, 1)
   l = tf.concat([c, l], 3)
   return l
   
   def add_transition(name, l, drop_rate, training):
   shape = l.get_shape().as_list()
   in_channel = shape[3]
   with tf.variable_scope(name) as scope:
   l = BatchNorm('bn1', l)
   l = tf.nn.relu(l)
   l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
   l = AvgPooling('pool', l, 2)
   #drop_out = tf.nn.dropout(l, keep_prob)
   
   l = tf.layers.dropout(l, rate = drop_rate, training=training)
   return l
   
   def dense_net(name):
   l = conv('conv0',image,  self.filters_init , 1)
   with tf.variable_scope('block1') as scope:
   
   for i in range(self.N):
   l = add_layer('dense_layer.{}'.format(i), l)
   l = add_transition('transition1', l ,self.drop_rate, self.train_or_test)
   
   with tf.variable_scope('block2') as scope:
   for i in range(self.N):
   l = add_layer('dense_layer.{}'.format(i), l)
   l = add_transition('transition2', l, self.drop_rate, self.train_or_test)
   
   with tf.variable_scope('block3') as scope:
   
   for i in range(self.N):
   l = add_layer('dense_layer.{}'.format(i), l)
   l = BatchNorm('bnlast', l)
   l = tf.nn.relu(l)
   l = GlobalAvgPooling('gap', l)
   logits = FullyConnected('linear', l, out_dim=2, nl=tf.identity)
   
   return logits
   
   def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
   with tf.name_scope('prediction_incorrect'):
   x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
   return tf.cast(x, tf.float32, name=name)
   
   logits = dense_net("dense_net") #map probabilities to real domain                                                  
   prob = tf.nn.softmax(logits, name='output')  #a generalization of the logistic function that "squashes" 
   factorbl = (self.class_0+self.class_1)/(2.0*self.class_0)
   factordl = (self.class_0+self.class_1)/(2.0*self.class_1)
   class_weights = tf.constant([factorbl, factordl])#factor,(1-factor)])#factor, 1.0-factor]) #dl 730 bl 1576          
   weights = tf.gather(class_weights, label)
   cost = tf.losses.sparse_softmax_cross_entropy(label, logits, weights=weights) #False posit                          
   cost = tf.reduce_mean(cost, name='cross_entropy_loss') #normalize                                                  
   wrong = prediction_incorrect(logits, label)
   # monitor training error                                                                                               
   add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
   # weight decay on all W                                                    
   wd_cost = tf.multiply(1e-3, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
   add_moving_summary(cost, wd_cost)
   add_param_summary(('.*/W', ['histogram']))   # monitor W                              
   self.cost = tf.add_n([cost, wd_cost], name='cost')
   return self.cost"""
   
   def _get_optimizer(self):
      lr = tf.get_variable('learning_rate', initializer=self.lr_init, trainable=False)
      tf.summary.scalar('learning_rate', lr)
      return tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999,epsilon=1e-08)#MomentumOptimizer(lr, 0.9, use_nesterov=True)
   
   non_depricated="""def optimizer(self):
   lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
   tf.summary.scalar('learning_rate', lr)
   return tf.train.AdamOptimizer(lr, beta1=0.88, beta2=0.999,epsilon=1e-08)"""

def get_data(train_or_test, shuffle = None, image_size = None, scale_size = None, scale = None, multi_crop = None, crop_per_case = None, unknown_dir = None, original_dir=None):
   isTrain = train_or_test == 'train'
   isVal = train_or_test == 'val'
   #ds = FakeData([[args.batch_size*10, 224, 224, 3], [args.batch_size*10]], 1000, random=False, dtype='uint32')
   ds = datapack.lymphoma2(train_or_test, image_size = image_size, scale_size = scale_size, scale = scale, multi_crop=multi_crop, crop_per_case = crop_per_case, shuffle = shuffle, dir = '../data', unknown_dir = unknown_dir,original_dir=original_dir)
   
   args.unique_samples = ds.unique_samples
   
   if train_or_test == 'train':
      args.class_0 = ds.class_0 
      args.class_1 = ds.class_1
   if args.class_weights:
      args.class_0 = args.class_weights[0]
      args.class_1 = args.class_weights[1]
   #pp_mean = ds.get_per_pixel_mean()
   #np.random.seed(None)
   aug_map_func = lambda dp: [dp[0], dp[1]]
   augmentors = []
   augmentor = None
   if isTrain:
      augmentors = [
         #and dividing by the standard deviation
         #datapack.HematoEAug((0.7, 1.3, np.random.randint(2**32-1), True)),
         datapack.NormStainAug(True),
         #imgaug.Flip(horiz=True),
         #imgaug.CenterPaste((args.scale_size, args.scale_size)),
      ]
      augmentor = imgaug.AugmentorList(augmentors)
      #ds = AugmentImageComponent(ds, augmentors, copy=True)
      aug_map_func=lambda dp: [augmentor.augment(dp[0].astype("uint8")),dp[1]]
   else:
      augmentors = [
         #imgaug.MapImage(lambda x: x - pp_mean),
         #imgaug.Brightness(20),
         #datapack.HematoEAug((0.7, 1.3, np.random.randint(2**32-1), True)),
         datapack.NormStainAug(True),
         #imgaug.CenterPaste((args.scale_size, args.scale_size)),
      ]
      ds = AugmentImageComponent(ds, augmentors, copy=True)
      
   
   print(">>>>>>> Data Set Size: ", ds.size())
   
   batch_size = None
   
   if train_or_test == 'test':
      print(">> Testing ", ds.size(), " images.")
      batch_size = ds.size()
   else:
      batch_size = args.batch_size
   
   if isTrain:
      ##ds = BatchData(ds, batch_size, remainder=not isTrain)
      ##ds = PrefetchData(ds, nr_prefetch = args.batch_size * args.num_gpu, nr_proc=8)
      print(">>>>> Setting up MultiThread Data Flow...")

      if args.mp == 2:
         ds = MultiProcessMapDataZMQ(ds,
                                     map_func=aug_map_func,
                                     nr_proc=mp.cpu_count())
      else:
         ## Multithreaded
         ds = MultiThreadMapData(ds,
                                 nr_thread= args.num_gpu,
                                 map_func = aug_map_func,
                                 buffer_size=batch_size*args.num_gpu)
         ds = PrefetchDataZMQ(ds, nr_proc = 1)
      
      ds = BatchData(ds, batch_size, remainder=not isTrain)
      print(">>> Done. |Data|/Batch Size = ", ds.size())
      return ds
   else:
      #ds = MapData(augmentor, aug_map_func)
      return BatchData(ds, batch_size, remainder=not isTrain)

def get_config(train_or_test, train_config = None):
   isTrain = train_or_test == 'train'
   log_dir = 'train_log/'+args.model_name+'-first%s-second%s-max%s' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
   logger.set_logger_dir(log_dir, action='n')
   
   drop_rate = 0.0
   dataset_train = 1
   dataset_val = None
   steps_per_epoch = 0
   
   # prepare dataset
   # dataflow structure [im, label] in parralel
   if isTrain:
      print(">>>>>> Loading training and validation sets")
      dataset_train = get_data('train', image_size = args.image_size, scale_size = args.scale_size,  scale = args.scale, multi_crop=args.multi_crop, crop_per_case = args.crop_per_case, shuffle = False)
      
      steps_per_epoch = dataset_train.size()/args.num_gpu if args.num_gpu > 1 else dataset_train.size()# = |data|/(batch size * num gpu)
      
      dataset_val = get_data('val', image_size = args.image_size, scale_size = args.scale_size, scale = args.scale, multi_crop=args.multi_crop, crop_per_case = args.crop_per_case, shuffle = False)

      drop_rate = 0.4
      # dense net
   
   print(" >>>>>>>>>> Steps Per Epoch: ", steps_per_epoch)
   print(">>>>>> Constructing Neural Network...")
   
   denseModel = Model(depth=args.depth, image_size = args.scale_size, lr_init = args.lr_init, kernels = args.kernels, kernel_size = args.kernel_size, expansion=args.expansion, class_0 = args.class_0, class_1 = args.class_1, drop_rate = drop_rate, train_or_test=isTrain)
   print(">>> Done.")
   
   if isTrain:
      print("Setting up training configuration: callbacks, validation checks and hyperparameter scheduling.")
      return TrainConfig(
         dataflow=dataset_train,
         callbacks=[
            MovingAverageSummary(), 
            ModelSaver(), # Record state graph at intervals during epochs
            InferenceRunner(input=dataset_val,
                            infs=[ScalarStats('cost'), ClassificationError()],
                            ),
            ScheduledHyperParamSetter('learning_rate',
                                      [(args.drop_0, args.scale_lr*args.lr_0),
                                       (args.drop_1,  args.scale_lr*args.lr_1)]),
            HyperParamSetterWithFunc('learning_rate',
                                     lambda e, x: x * 0.1 if e % 80 == 0 and e > args.drop_2 else x),# (1+e)/(2*20)
            #ScheduledHyperParamSetter('learning_rate',[(args.drop_0, args.scale_lr*args.lr_0), (args.drop_1,  args.scale_lr*args.lr_1), (args.drop_2,  args.scale_lr*args.lr_2), (args.drop_3,  args.scale_lr*args.lr_3)]), # denote current hyperparameters
            MergeAllSummaries()
         ],
         model=denseModel,
         session_creator = None,
         session_config = train_config,
         steps_per_epoch=steps_per_epoch,
         max_epoch=args.max_epoch,
      )
   else:
      """
      Predictive model configuration for testing 
      and classifying.
      """
      print(">>>>>> Constructing prediction variables.")
      return PredictConfig(
         model=denseModel,
         input_names= ['input', 'label'],#denseModel._get_inputs(),
         output_names=['output', 'train_error', 'cross_entropy_loss', 'input'],
      )

class predictModel:
   def __init__(self, config, data = None, image_size = None):
      self.test_data = data
      self.predictor = SimpleDatasetPredictor(config, self.test_data)
      
   def _get_inputs(self):
      return [InputDesc(tf.float32, [None, image_size, image_size, 3], 'input'),
              InputDesc(tf.int32, [None], 'label')
      ]
   
   def get_results(self):
      res = []
      all_res = []
      # Output for each datapoint in dataflow
      for prob, training_error, cross_entropy, img in self.predictor.get_result():
         res.append( ["element", prob, training_error, cross_entropy, img] )# tf.argmax(prob,1)
      for prob, training_error, cross_entropy, img in self.predictor.get_all_result():
         all_res.append(  ["element", prob , training_error, cross_entropy, img] )
      return res, all_res


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--model_name',type= str, default='MODEL',help="Name to prepend on model during training")
   parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') 
   parser.add_argument('--load', help='load model')
   parser.add_argument('--kernels',type=int, default=24, help='initial kernel channel depth. Number kernels.')
   parser.add_argument('--kernel_size',type=int, default=3, help='initial kernel window size')
   parser.add_argument('--expansion', type=int, default=12,help='expansion growth rate of kernels between layers per convolution')
   parser.add_argument('--drop_0',type=int, default=1, help='Epoch to drop learning rate to lr_0.')
   parser.add_argument('--drop_1',type=int, default=80, help='Epoch to drop learning rate to lr_1.')
   parser.add_argument('--drop_2',type=int, default=150,help='Epoch to drop learning rate to lr_2.')
   parser.add_argument('--drop_3',type=int, default=150,help='Epoch to drop learning rate to lr_3.')
   parser.add_argument('--lr_init',type=float, default=0.001,help='starting learning rate')
   parser.add_argument('--lr_0',type=float, default=0.0001,help='second learning rate')
   parser.add_argument('--lr_1',type=float, default=0.00001, help='first learning rate')             
   parser.add_argument('--lr_2',type=float, default=0.000001,help='second learning rate')
   parser.add_argument('--lr_3',type=float, default=0.0000001,help='third learning rate')
   parser.add_argument('--class_weights',type=str,help='comma seperated class weights e.g. class 0, class 1')
   parser.add_argument('--batch_size',type=int, default=4,help='batch size')
   parser.add_argument('--depth',type=int, default=13, help='The depth of densenet')
   parser.add_argument('--max_epoch',type= int, default=256,help='max epoch')
   parser.add_argument('--tot',type= str, default='train',help=" 'train' or 'test'")
   parser.add_argument('--out_dir',type= str, default='../data/Unknowns/predictions/',help="img out dir")
   parser.add_argument('--unknown_dir',type= str, default='../data/Unknowns/predictions/',help="unknown samples to classify")
   parser.add_argument('--save_sample',type= bool, default=False,help="boolean save originals")
   parser.add_argument('--original_dir',default=False,help="directory to save originals")
   parser.add_argument('--num_gpu',type= int,help="Number GPU to use if not specificaly assigned")
   parser.add_argument('--scale_lr',type= float,default=1.0,help="Scale learning rate factor (for distributed training)")
   parser.add_argument('--gpu_frac',type= float,default=0.99,help="Number GPU to use if not specificaly assigned")
   parser.add_argument('--mp',default=True,help="Whether or not to use parallel multiprocessing over or on GPU. 0 no, 1 yes. Default yes.")
   parser.add_argument('--nccl',default=True,help="Whether NCCL available for DGX like machines.")
   parser.add_argument('--class_0',type=int, default=0,help="number samples in class 0")
   parser.add_argument('--class_1',type=int, default=0,help="number samples in class 1")
   parser.add_argument('--multi_crop',type=int, default=4,help="number, if any, of crops to take from crop_per_case images. Average of crop classifications used in application")
   parser.add_argument('--image_size',type=int, default=448,help="original image size to scale to scale_size ")
   parser.add_argument('--scale_size',type=int, default=224,help="image size after scaling")
   parser.add_argument('--scale',type=int, default=2,help="factor of 224 to crop and scale down scale*(224,224)->(224,224)")
   parser.add_argument('--crop_per_case',type=int, default=40,help="number, more than 1, of images from each case to multi_crop. Average of crop classifications used in application")
   parser.add_argument('--unique_samples',type=int, help="Number of unique samples (excluding crops) used for averaging testing results")
   args = parser.parse_args()
   args.scale_size = args.image_size
   #if args.image_size != args.scale_size:
   #   args.scale = float(args.scale_size)/float(args.image_size)
   #   args.scale_size = int(args.image_size*args.scale)
   #   print( ">>>>> Scaling ", args.image_size, " to ", args.scale_size, " by factor ", args.scale)
   
   if args.class_weights:
      args.class_weights = [float(weight) for weight in args.class_weights.split(',')]
   
   if args.gpu or (not args.num_gpu):
      #from tensorpack.utils.gpu import get_num_gpu
      args.num_gpu = len(args.gpu.split(','))#max(get_num_gpu(),1)
      note = "Slurm assigns on DGX"
      print(" >>>> Hard assigning to all available gpu")
      #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
      #os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,range(args.num_gpu)))#args.gpu
   
   if not args.tot:
      args.tot == 'train'
      
   session_config = None
   if args.tot =='train':
      print("Using personal session config")
      #intra_op_parallelism_threads=num_cores,inter_op_parallelism_threads=num_cores,
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_frac, allow_growth=True)
      session_config = tf.ConfigProto(intra_op_parallelism_threads=args.num_gpu,inter_op_parallelism_threads=args.num_gpu, gpu_options=gpu_options, allow_soft_placement = True) #or session creator
      
   #if args.num_gpu:
   #   args.batch_size *= args.num_gpu
   
   nr_tower = 1
   if args.gpu or args.num_gpu:
      if args.num_gpu and (not args.gpu):
         print(">>>> Using "+str(args.num_gpu)+ " available GPU.")
         nr_tower = args.num_gpu
      else:
         nr_tower = len(args.gpu.split(','))
   
   config = get_config(args.tot, train_config=session_config)
   print(tf.test.is_gpu_available())                                          
   print(get_available_gpus())

   print("Net configured")
   
   if args.load:
      print(">>>> Loading stored model parameters.")
      # example args.load '/path/to/model/folder/model-xxxx'
      config.session_init = SaverRestore(args.load)
   
   if args.tot == 'train':
      if args.mp==0:
         print("using simple trainer")
         launch_train_with_config(config, SimpleTrainer())
      else:
         print("can use simple (mp=0) trainer multi gpu parameter server or replicated")
         print("for nccl as well as multiprocess distributed (mp=2) or multithreaded distributed (mp=else)")
         if args.gpu or bool(args.nccl) == False:
            print(">>>> Using "+str(args.num_gpu)+" available GPU parameter server.")
            launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(args.num_gpu))
         if args.num_gpu:
            print(">>>> Using "+str(args.num_gpu)+" available GPU for replicated training (nccl).")
            launch_train_with_config(config, SyncMultiGPUTrainerReplicated(args.num_gpu))
   else:
      # retrieve unknown from /data/Unknown/unknown_dir   optional save data to original_dir (creates if d.n.e.)
      # writes predictions (image and summary file) to path = '../data/Unknown/'+args.unknown_dir +'/predictions'
      
      data = get_data('test',image_size=args.image_size, scale_size = args.scale_size, scale = args.scale, multi_crop = args.multi_crop, crop_per_case = args.crop_per_case, shuffle = False,unknown_dir = args.unknown_dir, original_dir=args.original_dir)
      
      def classify_unknown(model_config, dp, args):
         predictor = predictModel(config, data = dp, image_size = args.scale_size)
         res, all_res = predictor.get_results()
         img_list = []
         prediction = []
         BL_count = 0
         DLBCL_count = 0
         tie = 0
         path = ''
         multi_crop_count = 0
         crop_per_case=1
         max_crop_met = 0
         
         crop_preds_bl = 0.0
         crop_preds_dl = 0.0
         if bool(args.multi_crop) != False:
            multi_crop_count = 0
            crop_per_case = min(args.unique_samples, args.crop_per_case)
            
         for i, op in enumerate(res[0][4]):
            prediction.append(res[0][1][i])
            img_list.append(op)
            
            #crop_preds_bl = 0.0
            #crop_preds_dl = 0.0
            
            if args.multi_crop==0 or args.crop_per_case == 0:
               max_crop_met += 1
            elif (i+1)/args.multi_crop >= args.crop_per_case:
               max_crop_met += 1
            
            path = '../data/Unknown/'+args.unknown_dir#predictions/'+args.out_dir+'_Predictions/'
            if not os.path.exists( os.path.join(path, 'predictions')):
               os.makedirs( os.path.join(path, 'predictions') )
            path =  os.path.join(path, 'predictions')
            
            im = Image.fromarray(np.asarray(op).astype('uint8'),'RGB')#.astype('uint8'))                                    
            im.save(path + '/' +str(i)+"_"+str(res[0][1][i])+'.jpeg')
            im.close()
            
            multi_crop_count += 1
            crop_preds_bl+=float(res[0][1][i][0])
            crop_preds_dl+=float(res[0][1][i][1])
            
            if multi_crop_count==args.multi_crop or max_crop_met > 0:
               if max_crop_met > 1:
                  multi_crop = 1.0
               else:
                  multi_crop = args.multi_crop if args.multi_crop != 0 else 1.0
               
               pred_bl = crop_preds_bl/float(multi_crop)
               pred_dl = crop_preds_dl/float(multi_crop)
               crop_preds_bl = 0.0
               crop_preds_dl = 0.0
               
               if pred_bl > 0.5:
                  BL_count += 1
               if pred_dl > 0.5:
                  DLBCL_count += 1
               if pred_bl == pred_dl:
                  tie += 1
               
               multi_crop_count = 0 if max_crop_met == 0 else -2
               crop_per_case -= 1
               
               path = '../data/Predictions/'+args.unknown_dir
               if not os.path.exists( path ):
                  os.makedirs( path )
               
               path = path+'/predictions'+args.unknown_dir+'.txt'
               if os.path.exists(path):
                  append_write = 'a'
                  #write results to file                   
               else:
                  append_write = 'w'
               prediction_file = open(path, append_write)
               prediction_file.write(str(pred_bl)+','+str(pred_dl)+"\n")
               prediction_file.close()
         
         prediction_summary = open( path, 'a')
         #'../data/Unknown/'+args.unknown_dir+'/predictions/predictions.txt', 'a')
         prediction_summary.write("\n")
         if tie:
            prediction_summary.write(str(BL_count)+','+str(DLBCL_count)+"\n")
            prediction_summary.write(str(tie)+"\n")
         else:
            prediction_summary.write(str(BL_count)+','+str(DLBCL_count)+"\n")
         prediction_summary.close()
         
      if args.batch_size == 1:
         classify_unknown(config, data, args)
      else:
         classify_unknown(config, data, args)

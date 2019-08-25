#!/home/sci/samlev/bin/bin/python3

#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t
#SBATCH --mem=90G
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:6

import numpy as np
import tensorflow as tf
import argparse
import os
import sys

# Personal Data Flow
sys.path.append(os.getcwd())
import datapack 

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_num_gpu

from PIL import Image


""" slurm gpu test 
"""
from tensorflow.python.client import device_lib
def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type == 'GPU']

#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(sess.run(hello))
#print(tf.test.is_gpu_available())
#print(get_available_gpus())



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
python3 denseNet_Lymphoma.py --gpu 0,1 (--num_gpu 4) (--load model-xxx) --drop_1 100 --drop_2 200 --depth 40 --max_epoch 368 --tot train
"""


BATCH_SIZE = 2

class Model(ModelDesc):
    def __init__(self, depth):
        super(Model, self).__init__()
        self.N = int((depth - 4)  / 3)
        self.growthRate = 12 

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0 - 1
        
        def conv(name, l, channel, stride):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
        def add_layer(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                c = BatchNorm('bn1', l)
                c = tf.nn.relu(c)
                c = conv('conv1', c, self.growthRate, 1)
                l = tf.concat([c, l], 3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                l = BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
                l = AvgPooling('pool', l, 2)
            return l


        def dense_net(name):
            l = conv('conv0',image,  16 , 1)
            with tf.variable_scope('block1') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition1', l)

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition2', l)

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=2, nl=tf.identity)

            return logits

        logits = dense_net("dense_net") #map probabilities to real domain

        prob = tf.nn.softmax(logits, name='output')  #a generalization of the logistic function that "squashes" a K-dim vector z  of arbitrary real values to a K-dim vector sigma( z ) of real values in the range [0, 1] that add up to 1.

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss') #normalize

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W


        
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test, unknown_dir = None, original_dir=None):
    isTrain = train_or_test == 'train'
    ds =  datapack.lymphoma2(train_or_test, dir = '../data', unknown_dir = unknown_dir,original_dir=original_dir) #dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            #and dividing by the standard deviation
            imgaug.CenterPaste((224, 224)),# ((910,910)), #
            #imgaug.RandomCrop((224, 224)), # ((900,900)), #
            imgaug.Flip(horiz=True),
            #imgaug.Brightness(20),
            #imgaug.Contrast((0.6,1.4)),
            #imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            #imgaug.MapImage(lambda x: x - pp_mean),
            #imgaug.Brightness(20),
            imgaug.CenterPaste((224, 224)),
            #imgaug.MapImage(lambda x: x - pp_mean),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

    
def get_config(train_or_test):
    isTrain = train_or_test == 'train'
    log_dir = 'train_log/'+args.model_name+'-first%s-second%s-max%s' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    # dataflow structure [im, label] in parralel
    dataset_train = get_data('train') 
    steps_per_epoch = dataset_train.size() #20
    dataset_test = get_data('test')

    # dense net
    denseModel = Model(depth=args.depth) #Use designed graph above, inheret from ModelDesc
    
    if isTrain:
        return TrainConfig(
            dataflow=dataset_train,
            callbacks=[
                ModelSaver(), # Record state graph at intervals during epochs
                InferenceRunner(dataset_train,
                                [ScalarStats('cost'), ClassificationError()]), #Compare to validation set
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)]) # denote current hyperparameters
            ],
            model=denseModel,
            steps_per_epoch=steps_per_epoch,
            max_epoch=args.max_epoch,
        )
    else:
        """
        Predictive model configuration for testing 
        and classifying.
        """
        return PredictConfig(
            model=denseModel,
            input_names= ['input', 'label'],#denseModel._get_inputs(),
            output_names=['output', 'train_error', 'cross_entropy_loss', 'input'],
        )

class predictModel:
    def __init__(self, config, data = None):
        self.test_data = data
        self.predictor = SimpleDatasetPredictor(config, self.test_data)

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),#  900,900,3], 'input'),#
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
   parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
   parser.add_argument('--load', help='load model')
   parser.add_argument('--drop_1',default=80, help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode
   parser.add_argument('--drop_2',default=150,help='Epoch to drop learning rate to 0.001')
   parser.add_argument('--depth',type=int, default=82, help='The depth of densenet')
   parser.add_argument('--max_epoch',type= int, default=256,help='max epoch')
   parser.add_argument('--tot',type= str, default='train',help=" 'train' or 'test'")
   parser.add_argument('--out_dir',type= str, default='../data/Unknowns/predictions/',help="img out dir")
   parser.add_argument('--unknown_dir',type= str, default='../data/Unknowns/predictions/',help="unknown samples to classify")
   parser.add_argument('--save_sample',type= bool, default=False,help="boolean save originals")
   parser.add_argument('--original_dir',default=False,help="directory to save originals")
   parser.add_argument('--num_gpu',type= int,help="Number GPU to use if not specificaly assigned")
   args = parser.parse_args()
   
   
   if args.gpu or get_num_gpu() == 1:
      note = "Slurm assigns on DGX"
      print(" >>>> hard assigning gpu")
      os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,range(get_num_gpu())))#args.gpu
         
   if not args.tot:
      args.tot == 'train'
           
   config = get_config(args.tot)
   if args.load:
      print(">>>> Loading model.")
      # example args.load '/path/to/model/folder/model-xxxx'
      config.session_init = SaverRestore(args.load)
      
   nr_tower = 1
   if args.gpu or args.num_gpu:
      if args.num_gpu:
         print(">>>> Using "+str(args.num_gpu)+ " available GPU.")
         nr_tower = args.num_gpu
      else:
         nr_tower = len(args.gpu.split(','))
      config.nr_tower = nr_tower
      
   if args.tot == 'train':
      num_gpu = max(get_num_gpu(),1) #len(args.gpu.split(','))#num_gpu#1#
      if get_num_gpu() == 1:
         launch_train_with_config(config, SimpleTrainer())
      else:
         if args.num_gpu:
            print(">>>> Using "+str(args.num_gpu)+" available GPU.")
            launch_train_with_config(config, SyncMultiGPUTrainer(args.num_gpu)) #SyncMultiGPUTrainerParameterServer
         else:
            print(">>>> Using all GPU.")
            launch_train_with_config(config, SyncMultiGPUTrainer(num_gpu))     
   else:
      data = get_data('test', unknown_dir = args.unknown_dir, original_dir=args.original_dir)
      predictor = predictModel(config, data)
      res, all_res = predictor.get_results()
      img_list = []
      prediction = []
      BL_count = 0
      DLBCL_count = 0
      tie = 0
      
      for i, op in enumerate(res[0][4]):
         prediction.append(res[0][1][i])
         img_list.append(op)
         
         path = '../data/Unknowns/predictions/'+args.out_dir+'_Predictions/'
         im = Image.fromarray(np.asarray(op).astype('uint8'),'RGB')#.astype('uint8'))                                    
         im.save('../data/Unknowns/predictions/'+args.out_dir+'_Predictions/'+str(i)+"_"+str(res[0\
][1][i])+'.jpeg')
         im.close()
         
         path = path+'/predictions.txt'
         if os.path.exists(path):
            append_write = 'a'
            #write results to file                                                                                          
         else:
            append_write = 'w'
         prediction_file = open(path, append_write)
         prediction_file.write(str(res[0][1][i][0])+','+str(res[0][1][i][1])+"\n")
         if res[0][1][i][0] > 0.5:
            BL_count += 1
         if res[0][1][i][1] > 0.5:
            DLBCL_count += 1
         if res[0][1][i][1] == 0.5:
            tie += 1
         prediction_file.close()
         prediction_file = open(path, append_write)
         if tie:
            prediction_file.write(str(BL_count)+','+str(DLBCL_count)+str(tie)+"\n")
         else:
           prediction_file.write(str(BL_count)+','+str(DLBCL_count)+"\n")
         prediction_file.close()
         """for i, op in enumerate(res[0][4]):
            prediction.append(res[0][1][i])
            img_list.append(op)

            path = '../data/Unknowns/predictions/'+args.out_dir+'_Predictions/'
            im = Image.fromarray(np.asarray(op).astype('uint8'),'RGB')#.astype('uint8'))
            im.save('../data/Unknowns/predictions/'+args.out_dir+'_Predictions/'+str(i)+"_"+str(res[0][1][i])+'.jpeg')
            im.close()

            path = path+'/predictions.txt'
            if os.path.exists(path):
               append_write = 'a'
            #write results to file
            else:
               append_write = 'w'
            prediction_file = open(path, append_write)
            prediction_file.write(str(res[0][1][i])+'\n')
            prediction_file.close()"""
            

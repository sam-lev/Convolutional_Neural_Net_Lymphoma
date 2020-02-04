#!/home/sci/samlev/bin/bin/python3

#SBATCH --time=21-00:00:00 # walltime, abbreviated by -t
#SBATCH --mem=120G
#SBATCH -o slurm-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e slurm-%j.err-%N # name of the stderr, using the job number (%j) and the first node (%N)
#SBATCH --gres=gpu:2

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
    ds =  datapack.lymphoma2(train_or_test, dir = '../data', unknown_dir = unknown_dir,original_dir=original_dir)
    #pp_mean = ds.get_per_pixel_mean()
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
    
    batch_size = None
    
    if train_or_test == 'test':
       print(">> Testing ", ds.size(), " images.")
       batch_size = ds.size()
    else:
       batch_size = args.batch_size
       
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain:
       ds = PrefetchData(ds, nr_prefetch = args.batch_size * args.num_gpu, nr_proc=50)
        
    return ds

    
def get_config(train_or_test, train_config = None):
    isTrain = train_or_test == 'train'
    log_dir = 'train_log/'+args.model_name+'-first%s-second%s-max%s' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    # dataflow structure [im, label] in parralel
    if isTrain:
       dataset_train = get_data('train') 
       steps_per_epoch = dataset_train.size() #20
       dataset_val = get_data('val')

    # dense net
    denseModel = Model(depth=args.depth) #Use designed graph above, inheret from ModelDesc
    # ! update here to build_graph() call bc _build_graph depricated
    
    if isTrain:
        return TrainConfig(
            dataflow=dataset_train,
            callbacks=[
                ModelSaver(), # Record state graph at intervals during epochs
                InferenceRunner(dataset_val,
                                [ScalarStats('cost'), ClassificationError()]), #Compare to validation set
                ScheduledHyperParamSetter('learning_rate',
                                          [(1, args.lr_0), (args.drop_1, args.lr_1), (args.drop_2, args.lr_2)]) # denote current hyperparameters
            ],
            model=denseModel,
            session_creator = None,
            session_config = train_config,
            #session_creator = train_sess_creator
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
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
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
   parser.add_argument('--drop_1',default=80, help='Epoch to drop learning rate to 0.01.')
   parser.add_argument('--drop_2',default=150,help='Epoch to drop learning rate to 0.001')
   parser.add_argument('--lr_0',type=float, default=0.1,help='second learning rate')
   parser.add_argument('--lr_1',type=float, default=0.01, help='first learning rate')             
   parser.add_argument('--lr_2',type=float, default=0.001,help='second learning rate')
   parser.add_argument('--batch_size',type=int, default=4,help='batch size')
   parser.add_argument('--depth',type=int, default=82, help='The depth of densenet')
   parser.add_argument('--max_epoch',type= int, default=256,help='max epoch')
   parser.add_argument('--tot',type= str, default='train',help=" 'train' or 'test'")
   parser.add_argument('--out_dir',type= str, default='../data/Unknowns/predictions/',help="img out dir")
   parser.add_argument('--unknown_dir',type= str, default='../data/Unknowns/predictions/',help="unknown samples to classify")
   parser.add_argument('--save_sample',type= bool, default=False,help="boolean save originals")
   parser.add_argument('--original_dir',default=False,help="directory to save originals")
   parser.add_argument('--num_gpu',type= int,help="Number GPU to use if not specificaly assigned")
   parser.add_argument('--gpu_frac',type= float,default=0.99,help="Number GPU to use if not specificaly assigned")
   parser.add_argument('--mp',default=True,help="Whether or not to use multi=process parallel threading over or on GPU. 0 for no, 1 for yes. default yes.")
   args = parser.parse_args()
   
   BATCH_SIZE = args.batch_size
   
   if args.gpu or (not args.num_gpu):
      #from tensorpack.utils.gpu import get_num_gpu
      args.num_gpu = len(get_available_gpus())#max(get_num_gpu(),1)
      note = "Slurm assigns on DGX"
      print(" >>>> Hard assigning to all available gpu")
      os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,range(args.num_gpu)))#args.gpu
         
   if not args.tot:
      args.tot == 'train'

   if args.tot =='train':
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_frac, allow_growth=True)
      #session_config = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)
      session_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement = True) #or session creator
   else:
      session_config = None

   
   config = get_config(args.tot, train_config=session_config) #train_sess_creator
   print(tf.test.is_gpu_available())                                                                                            
   print(get_available_gpus())
      
   if args.load:
      print(">>>> Loading model.")
      # example args.load '/path/to/model/folder/model-xxxx'
      config.session_init = SaverRestore(args.load)
      
   nr_tower = 1
   if args.gpu or args.num_gpu:
      if args.num_gpu and (not args.gpu):
         print(">>>> Using "+str(args.num_gpu)+ " available GPU.")
         nr_tower = args.num_gpu
      else:
         nr_tower = len(args.gpu.split(','))
      
   if args.tot == 'train':
      if args.num_gpu == 1 and not args.mp:
         launch_train_with_config(config, SimpleTrainer())
      else:
         if args.num_gpu:
            print(">>>> Using "+str(args.num_gpu)+" available GPU.")
            launch_train_with_config(config, SyncMultiGPUTrainerReplicated(args.num_gpu))     
   else:
      # retrieve unknown from /data/Unknown/unknown_dir   optional save data to original_dir (creates if d.n.e.)
      # writes predictions (image and summary file) to path = '../data/Unknown/'+args.unknown_dir +'/predictions/'
      data = get_data('test', unknown_dir = args.unknown_dir, original_dir=args.original_dir)

      def classify_unknown(model_config, dp, args):
         predictor = predictModel(config, dp)
         res, all_res = predictor.get_results()
         img_list = []
         prediction = []
         BL_count = 0
         DLBCL_count = 0
         tie = 0
      
         for i, op in enumerate(res[0][4]):
            prediction.append(res[0][1][i])
            img_list.append(op)
            
            path = '../data/Unknown/'+args.unknown_dir#predictions/'+args.out_dir+'_Predictions/'
            if not os.path.exists( os.path.join(path, 'predictions')):
               os.makedirs( os.path.join(path, 'predictions') )
            path =  os.path.join(path, 'predictions')
            
            im = Image.fromarray(np.asarray(op).astype('uint8'),'RGB')#.astype('uint8'))                                    
            im.save(path + '/' +str(i)+"_"+str(res[0][1][i])+'.jpeg')
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
         prediction_summary = open( '../data/Unknown/'+args.unknown_dir+'/predictions/predictions.txt', 'a')
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
            

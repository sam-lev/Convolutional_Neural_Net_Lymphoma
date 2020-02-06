import os
import sys
from LyCNN.datapack.IO.Convert_ZIDX import Convert_Dataset_ZIDX
from LyCNN.datapack.IO.VisusDataflow import ShowData
read_path = "/home/sam/Documents/PhD/Research/Convolutional_Neural_Net_Lymphoma/data/train"
write_path = "/home/sam/Documents/PhD/Research/Convolutional_Neural_Net_Lymphoma/data/train/train_idx"


if ((len(sys.argv) > 1) and bool(int(sys.argv[1]))):
    Convert_Dataset_ZIDX(read_path=read_path, write_path=write_path)

cat_sample ="/home/sam/anaconda3/lib/python3.7/site-packages/OpenVisus/datasets/cat/visus.idx"
sample = os.path.join(write_path,"batch_data_SP-19-5085_2.idx")

resolution = None if len(sys.argv) <= 2 else int(sys.argv[2])

ShowData(data=sample,load=True, resolution=resolution, print_attr=1)
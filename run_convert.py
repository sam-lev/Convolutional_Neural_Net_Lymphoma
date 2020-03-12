import os
import sys

from LyCNN.datapack.IO.Convert_ZIDX import Convert_Dataset_ZIDX
from LyCNN.datapack.IO.VisusDataflow import ShowData
from LyCNN.datapack.IO.VisusDataflow import ReadData
from LyCNN.datapack.lymphomaDataPack import lymphoma2ZIDX

base = "/home/sam/Documents/PhD/Research/Convolutional_Neural_Net_Lymphoma"#/home/sci/samlev/Convolutional_Neural_Net_Lymphoma"
unknown_path = os.path.join(base, "data", "train")#"Unknown")
#unknown_path, unknown_dirs, files_train = next(os.walk(os.path.join(unknown_path)))

test_path = "/home/sam/Documents/PhD/Research/Convolutional_Neural_Net_Lymphoma/data/train/train_idx"\
#"/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/data/test"#"/home/sam/Documents/PhD/Research/Convolutional_Neural_Net_Lymphoma/data/train"

read_path = unknown_path
write_path = os.path.join(read_path,read_path.split('/')[-1]+'_idx')

if ((len(sys.argv) > 1) and bool(int(sys.argv[1]))):
    #for dir in unknown_dirs:
    #    read_path = os.path.join(unknown_path,dir)
    #read_path = unknown_path
    #write_path = os.path.join(read_path,read_path.split('/')[-1]+'_idx')
    Convert_Dataset_ZIDX(read_path=read_path, write_path=write_path)
    print(" >> ")
    print(" >> ")
    print(" >>>>>    Done with Conversion. Testing results: ")
    print(" >> ")
    print(" >> ")

cat_sample ="/home/sam/anaconda3/lib/python3.7/site-packages/OpenVisus/datasets/cat/visus.idx"
sample = os.path.join(write_path,"batch_data_7-HE_32~0.idx")#batch_data_SP-19-5085_0.idx")# "batch_data_case_2_dlbcl_247.idx")


resolution = None if len(sys.argv) <= 2 else float(sys.argv[2])
if resolution == 0:
    resolution = None
#ShowData(data=sample,load=True, resolution=resolution, print_attr=1)
sample_data = ReadData(data=sample,load=True, resolution=resolution, print_attr=1).data
print(type(sample_data))
ShowData(data=sample_data, load=False, resolution=resolution, print_attr=1)

#last arg bool 0 or 1 to check dataflow
if ((len(sys.argv) > 2) and bool(int(sys.argv[3]))):
    dataset = lymphoma2ZIDX(train_or_test='', image_size=244, multi_crop=2, shuffle=False
                            ,dir=write_path, idx_filepath=write_path
                            ,idx=True,mode='r')
    print("batch size: ", dataset.size())
    show_first = 0
    for dp_ua in dataset:  # _unAug:
        print("Data shape OG: ", len(dp_ua), ' elm 0 size ', dp_ua[0].shape)
        ShowData(data = dp_ua[0], load=False, print_attr=1)

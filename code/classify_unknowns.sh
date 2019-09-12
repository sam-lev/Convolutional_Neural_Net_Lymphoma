#!/bin/bash

model = "$1"

source ~/.bashrc
export PATH="/home/sci/samlev/Convolutional_Neural_Net_Lymphoma/code:$PATH"

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir A --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir B --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir C --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir D --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir E --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir F --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir G --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir H --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir I --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir K --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir L --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir M --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir N --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir O --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir P --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir Q --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir R --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir S --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir J1 --depth 10 --class_weights 12270,13679

python3 multiThreadDistNNet.py --tot test --gpu 0,1 --batch_size 500 --model_name ep_134_unkown_A --load "$model" --unknown_dir J2 --depth 10 --class_weights 12270,13679

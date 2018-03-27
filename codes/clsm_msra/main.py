# coding = utf-8

import pandas as pd
import numpy as np
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import argparse
import os
import datetime
import traceback

from word_hashing import WordHashing
from model import CNN_clsm
from load_data import load_data

Data_path = '../../data/spark.csv'
Train_path = '../datas/train_set.csv'

parser = argparse.ArgumentParser(description='')
# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-kernel-num', type=int, default=300, help='number of each kind of kernel')
parser.add_argument('-kernel-size', type=str, default=3, help='comma-separated kernel size to use for convolution')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
args = parser.parse_args()


args.tri_letter_length = load_data(Data_path)
args.sementic_size = 128    

cnn = CNN_clsm(args)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()




train_data = pd.read(Train_path, encoding='gb18030', header=[])
train_iter= data.Iterator.splits(
                            train_data, 
                            batch_sizes=args.batch_size,
                            device=-1, 
                            repeat=False)


args.save_dir     = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))




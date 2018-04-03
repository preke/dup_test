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
from train import train
import sys
import csv
from get_embeddings import get_embeddings_and_split_datasets

csv.field_size_limit(sys.maxsize)

Data_path  = '../../data/spark.csv'
Train_path = './datas/train_set.csv'
Test_path  = './datas/test_set.csv'

parser = argparse.ArgumentParser(description='')
# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=10, help='how many steps to wait before saving [default:500]')
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

args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.sementic_size = 128    
embedding_dict, embedding_length = get_embeddings_and_split_datasets(Data_path, 'SPARK')
print('Embedding done. Vector length: %s.\n' %str(embedding_length))

TEXT = data.Field(sequential=True, use_vocab=True, batch_first=True)
train_data = data.TabularDataset(path=Train_path, 
                                 format='CSV',
                                 fields=[('query', TEXT), ('pos_doc', TEXT), ('neg_doc_1', TEXT), 
                                        ('neg_doc_2', TEXT), ('neg_doc_3', TEXT), ('neg_doc_4', TEXT),
                                        ('neg_doc_5', TEXT) ])

''''''
# train_data, vali_data = train_data.splits(split_ratio=0.9)
# TEXT.build_vocab(train_data, vali_data)
# train_iter, vali_iter = data.Iterator.splits(
#     (train_data, vali_data), 
#     batch_sizes=(args.batch_size, len(vali_data)), 
#     repeat=False)
''''''

TEXT.build_vocab(train_data)
print('Building vocabulary done. Vector length: %s.\n' %str(len(train_data)))
args.embedding_length = embedding_length
args.embedding_num    = len(TEXT.vocab)


train_iter = data.Iterator(train_data,
                          batch_size=args.batch_size,
                          device=0,
                          repeat=False)


word_vec_list = []
for idx, word in enumerate(TEXT.vocab.itos):
    if word in embedding_dict:
        vector = np.array(embedding_dict[word], dtype=float).reshape(1, embedding_length)
    else:
        vector = np.random.rand(1, args.embedding_length)
    word_vec_list.append(torch.from_numpy(vector))
wordvec_matrix = torch.cat(word_vec_list)
    
cnn       = CNN_clsm(args, wordvec_matrix)
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

''''''
# train(train_iter=train_iter, vali_iter=vali_iter, model=cnn, args=args)
''''''
train(train_iter=train_iter, vali_iter=None, model=cnn, args=args)



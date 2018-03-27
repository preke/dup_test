# coding = utf-8

import os
import pandas as pd
import numpy as np
import re
import random
import tarfile
import urllib
from torchtext import data
from datetime import datetime
import traceback
from sklearn.utils import shuffle  
from word_hashing import WordHashing

def load_data(data_path, prefix, neg_num=5):
    '''
        生成训练集和测试集

        prefix 为bug问题单的前缀：如 'SPARK-20000' 的 prefix 为 'SPARK'
        prefix 与数据集相对应

        正样本（标记为重复的pair）
        负样本（标记为非重复的pair（随机生成））
        正负样本比 1：neg_num（参数的最后一项）
    '''
    
    df = pd.read_csv(data_path, encoding = 'gb18030')
    df['Duplicate_null'] = df['Duplicated_issue'].apply(lambda x : pd.isnull(x))
    df_data              = df[df['Duplicate_null'] == False]
    df_field             = df_data[['Issue_id', 'Title', 'Duplicated_issue', 'Resolution']]
    df_field['dup_list'] = df_field['Duplicated_issue'].apply(lambda x: x.split(';'))
    data_set_index = []
    
    for i,r in df_field.iterrows():
        for dup in r['dup_list']:
            if int(r['Issue_id'].split('-')[1]) < int(dup.split('-')[1]):
                tmp_list = []
                if dup.startswith(prefix):
                    tmp_list.append(dup)
                for j in range(neg_num):
                    tmp = np.random.choice(df['Issue_id'].values)
                    if (tmp != r['Issue_id']) and (tmp != dup): 
                        tmp_list.append(tmp)
                data_set_index.append([r['Issue_id'], tmp_list])

    data_set_index = pd.DataFrame(data_set_index, columns=['query', 'doc_list'])
    data_set = pd.DataFrame([])
    data_set['query'] = data_set_index['query'].apply(\
        lambda x: list(df[df['Issue_id'] == x]['Title'])[0] if len(list(df[df['Issue_id'] == x]['Title'])) > 0 else '')
    data_set['doc_list'] = data_set_index['doc_list'].apply(\
        lambda xs: [(list(df[df['Issue_id'] == x]['Title'])[0] if len(list(df[df['Issue_id'] == x]['Title'])) > 0 else '') for x in xs])
    
    corpus = []
    for i, r in data_set.iterrows():
        corpus += r['query'].split(' ')
        for doc in r['doc_list']:
            corpus += doc.split(' ') 
    
    wh_instance = WordHashing(corpus)
    data_set['query_hashing_list'] = data_set['query'].apply(\
        lambda x: [wh_instance.hashing(i) for i in x.split(' ')])
    data_set['docs_hashing_list'] = data_set['doc_list'].apply(\
        lambda x: [[wh_instance.hashing(i) for i in doc.split(' ')] for doc in x])

    ratio = 0.8 # train set ratio
    data_set.head(int(ratio*len(data_set))).to_csv('../datas/train_set.csv', index=False)
    data_set.tail(int((1-ratio)*len(data_set))).to_csv('../datas/test_set.csv', index=False)



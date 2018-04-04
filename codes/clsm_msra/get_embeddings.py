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

def get_embeddings_and_split_datasets(data_path, prefix, neg_num=5):
    
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
                        while True:
                            tmp = np.random.choice(df['Issue_id'].values)
                            if (tmp != r['Issue_id']) and (tmp != dup): 
                                tmp_list.append(tmp)
                                break
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
    
    
    wh_instance      = WordHashing(corpus)
    embedding_length = 0
    embedding_dict   = {}
    for word in corpus:
        embedding_dict[word] = wh_instance.hashing(word)
        embedding_length     = len(wh_instance.hashing(word))

    data_set['pos_doc']   = data_set['doc_list'].apply(lambda x: x[0])
    data_set['neg_doc_1'] = data_set['doc_list'].apply(lambda x: x[1])
    data_set['neg_doc_2'] = data_set['doc_list'].apply(lambda x: x[2])
    data_set['neg_doc_3'] = data_set['doc_list'].apply(lambda x: x[3])
    data_set['neg_doc_4'] = data_set['doc_list'].apply(lambda x: x[4])
    data_set['neg_doc_5'] = data_set['doc_list'].apply(lambda x: x[5])

    data_set = data_set[['query', 'pos_doc', 'neg_doc_1', 'neg_doc_2',
                        'neg_doc_3', 'neg_doc_4', 'neg_doc_5']]


    train_set = data_set.head(int(0.8*len(data_set)))
    train_set = train_set.head(int(0.9*len(data_set)))
    vali_set  = train_set.tail(int(0.1*len(data_set)))
    test_set  = data_set.tail(int(0.2*len(data_set)))
    
    test_pairs = []
    for i,r in test_set.iterrows():
        test_pairs.append([r['query'], r['pos_doc'],   '1'])
        test_pairs.append([r['query'], r['neg_doc_1'], '0'])
        test_pairs.append([r['query'], r['neg_doc_2'], '0'])
        test_pairs.append([r['query'], r['neg_doc_3'], '0'])
        test_pairs.append([r['query'], r['neg_doc_4'], '0'])
        test_pairs.append([r['query'], r['neg_doc_5'], '0'])

    test_pairs = pd.DataFrame(test_pairs, columns=['query', 'doc', 'label'], index=False)
    
    train_set.to_csv('./datas/train_set.csv', index=False)
    vali_set.to_csv('./datas/vali_set.csv', index=False)
    test_pairs.to_csv('./datas/test_set.csv', index=False)
    
    return embedding_dict, embedding_length



    
# coding = utf-8
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from math import sqrt
from math import log
import pandas as pd
import numpy as np
import logging
import logging.config
import traceback
import datetime

# from load_data import load_data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
config_file = 'logging.ini'
logging.config.fileConfig(config_file, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class Preprocess(object):
    def __init__(self):
        self.english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'']
        self.stop = set(stopwords.words('english'))

    def punctuate(self, text):
        ans = ""
        for letter in text:
            if letter in self.english_punctuations:
                ans += ' '
            else:
                ans += letter
        return ans

    def stem_and_stop_removal(self, text):
        try:
            text = self.punctuate(text)
            word_list = word_tokenize(text)
            lancaster_stemmer = LancasterStemmer()
            word_list = [lancaster_stemmer.stem(i) for i in word_list]
            word_list = [i for i in word_list if i not in self.stop]
            return word_list
        except:
            print text
            return []

class Vectorization(object):
    def __init__(self, df):
        self.word_set = set([])
        for desc in df:
            for word in desc:
                if not (word in self.word_set):
                    self.word_set.add(word)
        logger.info("Altogether %d words in the word_set." % len(self.word_set))

    def vectorize(self, desc):
        temp_list = []
        for word in self.word_set:
            if word in desc:
                cnt = 0
                for item in desc:
                    if item == word:
                        cnt += 1
                temp_list.append(float(3+ 2*log(cnt)))
            else:
                temp_list.append(0)
        return temp_list

def check(vec):
    for i in vec:
        if i != 0:
            return False
    return True

if __name__ == '__main__':
    # load_data('spark.csv')
    pos = pd.read_csv('hadoop/pos.csv', names = ['Issue_1', 'Issue_2'])
    neg = pd.read_csv('hadoop/neg.csv', names = ['Issue_1', 'Issue_2'])

    preprocess = Preprocess()
    logger.info('Title')

    pos['Issue1_list'] = pos['Issue_1'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    pos['Issue2_list'] = pos['Issue_2'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    neg['Issue1_list'] = neg['Issue_1'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    neg['Issue2_list'] = neg['Issue_2'].apply(lambda x : preprocess.stem_and_stop_removal(x))

    corpus = pd.concat([pos['Issue1_list'], pos['Issue2_list'], neg['Issue1_list'], neg['Issue2_list']])
    vectorization = Vectorization(corpus)

    pos['vec1'] = pos['Issue1_list'].apply(lambda x : vectorization.vectorize(x))
    pos['vec2'] = pos['Issue2_list'].apply(lambda x : vectorization.vectorize(x))
    neg['vec1'] = neg['Issue1_list'].apply(lambda x : vectorization.vectorize(x))
    neg['vec2'] = neg['Issue2_list'].apply(lambda x : vectorization.vectorize(x))


    pos['vec_null'] = pos['vec1'].apply(lambda x : check(x))
    pos = pos[pos['vec_null'] == False]
    pos['vec_null'] = pos['vec2'].apply(lambda x : check(x))
    pos = pos[pos['vec_null'] == False]

    neg['vec_null'] = neg['vec1'].apply(lambda x : check(x))
    neg = neg[neg['vec_null'] == False]
    neg['vec_null'] = neg['vec2'].apply(lambda x : check(x))
    neg = neg[neg['vec_null'] == False]

    pos['label'] = '1'
    neg['label'] = '0'

    ratio  = [0.7, 0.1, 0.2]
    train_set = pd.concat([pos.head(len(pos) * ratio[0]), neg.head(len(neg) * ratio[0])])
    vali_set  = pd.concat([pos.head(len(pos) * ratio[1]), neg.head(len(neg) * ratio[1])])
    test_set  = pd.concat([pos.head(len(pos) * ratio[2]), neg.head(len(neg) * ratio[2])])
    # train_set = train_set.sample(1)
    # test_set = test_set.sample(1)

    # train:
    logger.info('Training...')
    query_list = pd.concat([train_set['vec1'], train_set['vec2']])
    sim_matrix = [[0]*len(train_set)]*len(train_set)
    for i in len(query_list):
        if i['label'] == '1':
            sim_matrix[i][i+len(train_set)] = cosine_similarity(train_set['vec1'], train_set['vec2'])

    cluster_label = SpectralClustering(n_clusters = 10, affinity = 'precomputed').fit_predict(np.array(sim_matrix))
    # test

    # print train_set.head()
    test_set['similarity'] = test_set.apply(lambda row: cosine_similarity(row['vec1'], row['vec2'])[0][0], axis = 1)
    for i,r in test_set.iterrows():
        a = r['vec1']
        b = r['vec2']
        top_a = 0
        temp_a = 0
        for j in range(len(query_list)):
            if cosine_similarity(a,query_list[j]) > temp_a:
                top_a = j
                temp_a = cosine_similarity(a,query_list[j])
        top_b = 0
        temp_b = 0
        for j in range(len(query_list)):
            if cosine_similarity(b,query_list[j]) > temp_b:
                top_b = j    
                temp_b = cosine_similarity(b, query_list[j])
        if cluster_label[top_a] == cluster_label[top_b]:
            ans.append('1')
        else:
            ans.append('0')

    # mean = train_set[train_set['label'] == '1']['similarity'].mean()
    # print mean
    # --- cut off 0
    # test:
    test_set['new_label'] = ans
    acc = test_set.apply(lambda x: 1 if x['label'] == x['new_label'] else 0, axis = 1).sum()/float(len(test_set))
    p = len(test_set[(test_set['new_label'] == '1') & (test_set['label'] == '1')]) / float(len(test_set[test_set['new_label'] == '1']))
    r = len(test_set[(test_set['new_label'] == '1') & (test_set['label'] == '1')]) / float(len(test_set[test_set['label'] == '1']))
    f1 = 2*(p*r)/((p+r))

    print 'acc:%f,\n p:%f,\n r:%f,\n f1:%f\n' %(acc, p, r, f1)


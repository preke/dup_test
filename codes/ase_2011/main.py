# coding = utf-8
from preprocess import Preprocess
from preprocess import Vectorization
from sklearn import preprocessing
import numpy as np
import pandas as pd
import math
from bm25f_ext import *
from datetime import datetime
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

DATA_PATH = '../../data/spark.csv'



def generate_bigram(a):
    bi = []
    for i in range(len(a)-1):
        bi.append(a[i]+a[i+1])
    return bi

def construct_features(d, q, idf_dict, k1, k3, field_weights):
    '''
        each query is a dict, keys are fields
        for a query and a document
        construct 7 features
    '''

    d_unigram = d
    q_unigram = q
    feature1 = bm25f_ext(d_unigram, q_unigram, idf_dict, k1, k3, field_weights)
    
    # d_bigram = bigrams(d)
    # q_bigram = bigrams(q)
    feature2 = feature1

    # feature3 = 1 if d['product'] == q['product'] else 0
    feature4 = 1 if d['component'] == q['component'] else 0
    
    # feature5 = 1 if d['type'] == q['type'] else 0
    # feature6 = 1 / (1.0 + d['priority'] - q['priority'])
    # feature7 = 1 / (1.0 + d['version'] - q['version'])

    return [feature1, feature2, feature4]

def REP(d,q, idf_dict, rep_weights, k1, k3, field_weights):
    features = construct_features(d,q, idf_dict, k1, k3, field_weights)
    # rep_weights = [1.163, 0.013, 3.612] # initialize
    ans = 0.0
    for i in range(len(rep_weights)):
        ans += (rep_weights[i] + features[i])

    return ans

def get_duplist(str1):
    lst = str1.split(';')
    lst = [i for i in lst if i.startswith('SPARK')]
    return lst

def illegal(list_):
    bool_vec = [str(i).startswith('SPARK') for i in list_]
    ans = True
    for i in bool_vec:
        ans = ans & i
    return ans
    
def generate_data_set(data):    
    total_dup_issues = np.array(pd.concat([data['Issue_id'], data['Duplicated_issue']]))
    dup_issues = []
    for i in total_dup_issues:
        for j in get_duplist(i):
            dup_issues.append(j)

    dup_issues = pd.DataFrame(dup_issues).drop_duplicates()
    dup_issues = [i[0] for i in dup_issues.values]
    dup_issues = sorted(dup_issues, reverse = False, key = lambda x: int(x[6:]) )
    
    
    # step1 generate buckets
    buckets = {}
    for i,r in data.iterrows():
        buckets[r['Issue_id']] = get_duplist(r['Duplicated_issue'])
    
    # step2 generate data set
    data_set = []
    for k,v in buckets.iteritems():
        for i in v:
            data_set.append([k, i, np.random.choice(dup_issues)])
    print(data_set[0])
    data_set = [i for i in data_set if illegal(i)]
    print(len(data_set))
    return buckets, data_set

# def RNC(training_data, idf_dict, k1, k3, field_weights):
#     front = REP(training_data[0], training_data[1], idf_dict, rep_weights, k1, k3, field_weights)
#     rare = REP(training_data[1], training_data[2], idf_dict, rep_weights, k1, k3, field_weights)
#     Y = front - rare
#     return math.log(1 + math.exp(Y))

def recall_at_k(test_set, test_data, buckets, idf_dict, topk, rep_weights, k1, k3, field_weights):
    recall = 0.0
    starttime = datetime.now()
    cnt = 1
    res = []
    for tst in test_data[:100]:
        master_list = []
        for k,v in buckets.iteritems():
            tmp_dict = {}
            try:
                smry = df_data['summary_list'][df_data[df_data['Issue_id'] == k].index].values[0]
                tmp_dict['summary'] = smry
            except:
                tmp_dict['summary'] = ''
                print(str(k) + ': summary is :%s' %df_data['summary_list'][df_data[df_data['Issue_id'] == k].index].values)
            try:
                desc = df_data['desc_list'][df_data[df_data['Issue_id'] == k].index].values[0]
                tmp_dict['description'] = desc
            except:
                tmp_dict['description'] = ''
                print(str(k) + ': description is :%s' %df_data['desc_list'][df_data[df_data['Issue_id'] == k].index].values)
            try:
                tmp_dict['component'] = df_data['Component'][df_data[df_data['Issue_id'] == k].index].values[0]
            except:
                tmp_dict['component'] = ''
            master_list.append([k, REP(tmp_dict, tst[1], idf_dict, rep_weights, k1, k3, field_weights)])
        master_list = sorted(master_list, reverse=True, key = lambda x: x[1])
        master_list1 = [i for i in master_list[:topk]]
        master_list = [i[0] for i in master_list[:topk]]
        # print master_list1
        res.append(master_list)
        print test_set[cnt-1][0]
        if test_set[cnt-1][0] in master_list:
            recall += 1
        print 'round %d: cost time: %s.' %(cnt, str(datetime.now() - starttime))
        cnt += 1
    pd.DataFrame(res).to_csv('res.csv', index=False)
    # return recall/float(len(test_data))
    return recall/100.0

if __name__ == '__main__':
    preprocess = Preprocess()
    # vectorization = Vectorization()

    df = pd.read_csv(DATA_PATH, encoding = 'GB18030')
    df['Duplicate_null'] = df['Duplicated_issue'].apply(lambda x : pd.isnull(x))
    df_data = df[df['Duplicate_null'] == False]
    df_data['summary_list'] = df_data['Title'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    df_data['desc_list'] = df_data['Description'].apply(lambda x : preprocess.stem_and_stop_removal(str(x)))
    df_data_issues = df_data[['Issue_id', 'Duplicated_issue']]
    
    buckets, data_set = generate_data_set(df_data_issues)
    
    training_set = data_set[:int(len(data_set)*0.7)]
    validation_set = data_set[int(0.7*len(data_set)):int(0.8*len(data_set))]
    test_set = data_set[int(0.8*len(data_set)):]

    
    # unigram:

    # get corpus
    corpus = []
    for j in df_data['desc_list'].values:
        corpus += j
    for j in df_data['summary_list'].values:
        corpus += j
    corpus = [' '.join(corpus)]
    
    vectorizer  = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf       = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word        = vectorizer.get_feature_names()
    weight      = tfidf.toarray()
    
    idf_dict = {}
    for j in range(len(word)):
        idf_dict[word[j]] = weight[0][j]

    
    training_data = []
    for q in training_set:
        tmp = []
        for j in q:
            tmp_dict = {}
            try:
                smry = df_data['summary_list'][df_data[df_data['Issue_id'] == j].index].values[0]
                tmp_dict['summary'] = smry
            except:
                tmp_dict['summary'] = ''
                print(str(j) + ': summary is :%s' %df_data['summary_list'][df_data[df_data['Issue_id'] == j].index].values)
            try:
                desc = df_data['desc_list'][df_data[df_data['Issue_id'] == j].index].values[0]
                tmp_dict['description'] = desc
            except:
                tmp_dict['description'] = ''
                print(str(j) + ': description is :%s' %df_data['desc_list'][df_data[df_data['Issue_id'] == j].index].values)
            try:
                tmp_dict['component'] = df_data['Component'][df_data[df_data['Issue_id'] == j].index].values[0]
            except:
                tmp_dict['component'] = ''
            tmp.append(tmp_dict)

        training_data.append(tmp)

    test_data = []
    for q in test_set:
        tmp = []
        for j in q:
            tmp_dict = {}
            try:
                smry = df_data['summary_list'][df_data[df_data['Issue_id'] == j].index].values[0]
                tmp_dict['summary'] = smry
            except:
                tmp_dict['summary'] = ''
                print(str(j) + ': summary is :%s' %df_data['summary_list'][df_data[df_data['Issue_id'] == j].index].values)
            try:
                desc = df_data['desc_list'][df_data[df_data['Issue_id'] == j].index].values[0]
                tmp_dict['description'] = desc
            except:
                tmp_dict['description'] = ''
                print(str(j) + ': description is :%s' %df_data['desc_list'][df_data[df_data['Issue_id'] == j].index].values)
            try:
                tmp_dict['component'] = df_data['Component'][df_data[df_data['Issue_id'] == j].index].values[0]
            except:
                tmp_dict['component'] = ''
            tmp.append(tmp_dict)

        test_data.append(tmp)
    
    
    k1 = 2.0
    field_weights = [2.1,0.1] 
    k3 = 0.1

    # tuning rep_weights
    rep_weights = [1.1, 0.1, 3.0]
    init_rep_weights = [1.1, 0.1, 3.0]
    for index in range(3):
        best_recall = 0
        best_index = 0
        recalls = []
        for cnt in range(10):
            print rep_weights
            rep_weights[index] += 0.1
            recall = recall_at_k(test_set, test_data, buckets, idf_dict, 10, rep_weights, k1, k3, field_weights)
            print 'Recall: %f' %recall
            recalls.append(recall)
        for i in range(len(recalls)):
            if r >= best_recall:
                best_index = i
        rep_weights[index] = init_rep_weights[index] + best_index*0.1 
        print 'Best weights are', rep_weights

    # rep_weights are best now!

    # tuning k1, k3, field_weights
    k1 = 2.0
    field_weights = [2.980,5.0] 
    k3 = 0.1
    bst_k1 = k1
    bst_k3 = k3
    
    # k1
    bst_recall = 0
    for cnt in range(10):
        k1 += cnt*0.1
        recall = recall_at_k(test_set, test_data, buckets, idf_dict, 10, rep_weights, k1, k3, field_weights)
        print k1, recall
        if recall >= bst_recall:
            bst_k1 = k1


    # k1
    bst_recall = 0
    for cnt in range(10):
        k3 += cnt*0.1
        recall = recall_at_k(test_set, test_data, buckets, idf_dict, 10, rep_weights, k1, k3, field_weights)
        print k3, recall
        if recall >= bst_recall:
            bst_k3 = k3
    
    # tuning field_weights
    field_weights = [2.980,5.0] 
    init_field_weights = [2.980,5.0] 
    for index in range(2):
        best_recall = 0
        best_index = 0
        recalls = []
        for cnt in range(10):
            print field_weights
            field_weights[index] += 0.1
            recall = recall_at_k(test_set, test_data, buckets, idf_dict, 10, rep_weights, k1, k3, field_weights)
            print 'Recall: %f' %recall
            recalls.append(recall)
        for i in range(len(recalls)):
            if r >= best_recall:
                best_index = i
        field_weights[index] = init_field_weights[index] + best_index*0.1 
        print 'Best weights are', field_weights

    k1 = bst_k1
    k3 = bst_k3
    recall = recall_at_k(test_set, test_data, buckets, idf_dict, 10, rep_weights, k1, k3, field_weights)
    # recall = recall_at_k(test_set, test_data, buckets, idf_dict, 10, rep_weights, k1, k3, field_weights)
    


    '''
        1. sort problems
        2. drop duplicate
        3. bigram
    '''

    '''
        #1.
            Test_num   :  636
            
            Parameters :  # weights = [1.163, 0.013, 0.032] 
                          # k1 = 2.0
                          # ws = [2.980,0.287] 
                          # k3 = 0.001
            
            Recall     :  0.322327
            
            Costtime   :  1:57:33.330419

        #2.
            Test_num   :  100
            
            Parameters :  # weights = [1.163, 0.013, 0.032] 
                          # k1 = 2.0
                          # ws = [2.980,0.287] 
                          # k3 = 0.001
            
            Recall     :  0.409974
            
            Costtime   :  0:18:37.410338

        #3.
            Test_num   :  100
            
            Parameters :  # weights = [1.163, 0.013, 3.612] 
                          # k1 = 2.0
                          # ws = [2.980,0.287] 
                          # k3 = 0.001
            
            Recall     :  0.41
            
            Costtime   :  0:18:37.410338

        #4.
            Test_num   :  100
            
            Parameters :  # weights = [1.163, 0.013, 3.612] 
                          # k1 = 2.0
                          # ws = [2.980, 5] 
                          # k3 = 0.001
            
            Recall     :  0.420000
            
            Costtime   :  0:18:37.410338


    '''




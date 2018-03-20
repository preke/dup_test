# coding = utf-8
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from math import sqrt
from math import log
import pandas as pd
import logging
import logging.config
import traceback
import datetime
from load_data import load_data
from sklearn.metrics.pairwise import cosine_similarity

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
        text = self.punctuate(text)
        word_list = word_tokenize(text)
        lancaster_stemmer = LancasterStemmer()
        word_list = [lancaster_stemmer.stem(i) for i in word_list]
        word_list = [i for i in word_list if i not in self.stop]
        return word_list

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
                temp_list.append(float(1+ log(cnt)))
            else:
                temp_list.append(0)
        return temp_list


def Cosine_simlarity(vec1, vec2):
    return cosine_similarity(vec1, vec2)

    # up = 0.0
    # down = 0.0
    # down_1 = 0.0
    # down_2 = 0.0
    # for i in range(len(vec1)):
    #     up += (vec1[i] * vec2[i])
    # for i in range(len(vec1)):
    #     down_1 += (vec1[i] * vec1[i])
    #     down_2 += (vec2[i] * vec2[i])
    # down = sqrt(down_1) * sqrt(down_2)
    # return float(up/down)

def Jacarrd(vec1, vec2):
    up = float(len(set(vec1) & set(vec2)))
    down = float(len(set(vec1) | set(vec2)))
    return up/down

def Dice(vec1, vec2):
    up = float(len(set(vec1) & set(vec2))) * 2.0
    down = float(len(set(vec1)) + len(set(vec2)))
    return up/down


def check(vec):
    for i in vec:
        if i != 0:
            return False
    return True

# def Calculate_similarity(df_data, time_frame, max_top_list):

#     logger.info(('Calculate cosine similarity begin... time_frame is %d, max_top_list = top %d') % (time_frame, max_top_list))
#     Cosine_matrix = []
#     for index_1, row_1 in df_data.iterrows():
#         cosine_vec = []
#         start = pd.to_datetime(row_1['Created_time'])
#         end = pd.to_datetime(pd.to_datetime(row_1['Created_time']) - datetime.timedelta(days = time_frame))
#         for index_2, row_2 in df_data.iterrows():
#             if (pd.to_datetime(row_2['Created_time']) <= start) & (pd.to_datetime(row_2['Created_time']) >= end):
#                 cosine_vec.append((row_2['Issue_id'], Cosine_simlarity(row_1['vec'], row_1['vec'])))
#             else:
#                 pass
#         cosine_vec = sorted(cosine_vec, key=lambda x:x[1], reverse=True)
#         while len(cosine_vec) < max_top_list:
#             cosine_vec.append(-1)
#         Cosine_matrix.append(cosine_vec)
#     pd.DataFrame(Cosine_matrix).to_csv('../res/Cosine_matrix'+time_frame+'.csv')
#     logger.info('Calculate cosine similarity end...')


if __name__ == '__main__':
    # load_data('spark.csv')
    pos = pd.read_csv('pos.csv', names = ['Issue_1', 'Issue_2'])
    neg = pd.read_csv('neg.csv', names = ['Issue_1', 'Issue_2'])
    
    
    preprocess = Preprocess()
    logger.info('Title')
    
    pos['Issue1_list'] = pos['Issue_1'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    pos['Issue2_list'] = pos['Issue_2'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    neg['Issue1_list'] = neg['Issue_1'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    neg['Issue2_list'] = neg['Issue_2'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    
    corpus = pd.concat([pos['Issue1_list'], pos['Issue2_list'], neg['Issue1_list'], neg['Issue2_list']])
    vectorization = Vectorization(corpus)
    
    pos['vec1'] = pos['vec1'].apply(lambda x : vectorization.vectorize(x))
    pos['vec2'] = pos['vec2'].apply(lambda x : vectorization.vectorize(x))
    neg['vec1'] = neg['vec1'].apply(lambda x : vectorization.vectorize(x))
    neg['vec2'] = neg['vec2'].apply(lambda x : vectorization.vectorize(x))
    
    pos['vec_null'] = pos['vec1'].apply(lambda x : check(x))
    pos = pos[pos['vec_null'] == False]
    pos['vec_null'] = pos['vec2'].apply(lambda x : check(x))
    pos = pos[pos['vec_null'] == False]

    neg['vec_null'] = neg['vec1'].apply(lambda x : check(x))
    neg = neg[neg['vec_null'] == False]
    neg['vec_null'] = neg['vec2'].apply(lambda x : check(x))
    neg = neg[neg['vec_null'] == False]
    
    ratio  = [0.7, 0.1, 0.2]
    train_set = pd.concat([pos.head(len(pos) * ratio[0]), neg.head(len(neg) * ratio[0])])
    vali_set  = pd.concat([pos.head(len(pos) * ratio[1]), neg.head(len(neg) * ratio[1])])
    test_set  = pd.concat([pos.head(len(pos) * ratio[2]), neg.head(len(neg) * ratio[2])])

    # train:
    logger.info('Training...')
    train_set['similarity'] = train_set.apply(lambda row: cosine_similarity(row['vec1'], row['vec_2']))
    mean = train_set[train_set['label'] == '1']['similarity'].agg(np.mean)
    print mean

    # test:
    logger.info('Test...')
    
    # df = pd.read_csv(open('spark.csv', 'rU'))
    
    
    # df['Duplicate_null'] = df['Duplicated_issue'].apply(lambda x : pd.isnull(x))
    # df_data = df[df['Duplicate_null'] == False]
    # preprocess = Preprocess()
    # logger.info('Title')
    # df_data['Desc_list'] = df_data['Title'].apply(lambda x : preprocess.stem_and_stop_removal(x))
    
    # # logger.info('Title with Description')
    # # df_data['Title_Desc'] = df.apply(lambda row: row['Title'] + ' ' + row['Description'], axis = 1)
    # # df_data['Desc_list'] = df_data['Title_Desc'].apply(lambda x : preprocess.stem_and_stop_removal(x))

    # vectorization = Vectorization(df_data['Desc_list'])
    # df_data['vec'] = df_data['Desc_list'].apply(lambda x : vectorization.vectorize(x))
    # df_data['vec_null'] = df_data['vec'].apply(lambda x : check(x))
    # df_data = df_data[df_data['vec_null'] == False]
    # Calculate_similarity(df_data, 500, 15)
    



    
    




















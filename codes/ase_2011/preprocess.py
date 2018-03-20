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
from sklearn.metrics.pairwise import cosine_similarity

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

# baseline models
from dumb_predictors import MeanRegressor, ModeClassifier

# NLP
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Regression Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Testing and optimization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.regression import mean_absolute_error, mean_squared_error, r2_score

# import module
from pipeline import remove_accents
from pipeline import tokenize
from pipeline import vectorize
from model import Model
from EDA import EDA

import pandas as pd
import numpy as np
import pickle
import datetime as dt
import glob
import sys
import os.path


class Detector(object):
    """
    A class to predict whether a news article is fake.
    """
    def __init__(self, model_obj, eda_obj):
        # self.model = None
        self.article = None
        self.corpus = []
        self.output = ''
        self.sw = eda_obj.sw
        self.tv = eda_obj.tv


        # self.df = None
        # self.types = None
        # self.bow = None
        # self.tf = None
        # self.tfidf = None
        # self.type_word_lst = {}
        # self.W = None
        # self.H = None
        # self.H_df = None
        # self.cv = None
        # self.tv = None
        # self.sw = None


    def inp(self):
        """
        take input
        """
        self.article = input("Enter your news article: ")

        return self

    def get_eda_atr(self):
        """get attributes from eda objects"""
        self.sw = eda_object.sw
        self.tv = eda_object.tv
        self.bow = eda_object

    def run(self, trained_model, stopwords):
        """
        run the detector, output the prediction
        """
        self.inp()
        self.corpus = self.get_corpus(self.article, stopwords)
        vector = trained_model.tv.transform(self.corpus)
        pred = trained_model.predict(vector)
        if pred == 1:
            self.output = 'fake'
        else:
            self.output = 'not fake'

        print('This news is {}.'.format(output))

    def get_corpus(self, article, sw):
        """
        Turn a article into a corpus
        :param article: str
        :return: list of corpus
        """
        article = pd.Series(article)
        tokens = tokenize(article, sw)[0]
        corpus = [' '.join(token for token in tokens)]

        return corpus

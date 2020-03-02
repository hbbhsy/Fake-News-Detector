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
import boto3
import datetime as dt
import glob
import sys
import os.path


class Detector(object):
    """
    Detector - Fake News Detect
    A class to create predictions on the reliability of news articles.
    """
    def __init__(self):
        self.article = None
        self.corpus = []
        self.sw = None
        self.model = None

    def get_model(self):
        """Load trained model from pickle, stored on AWS S3"""
        # loading pickled model from AWS S3
        print('Loading the detector...')
        s3 = boto3.resource('s3')
        bucket = 'fakenewscorpus'
        data_key = ''
        data_location = 's3://{}/{}'.format(bucket, data_key)
        my_pickle = pickle.loads(s3.Bucket(bucket).Object(data_key).get()['Body'].read())
        with open('my_pickle','rb') as f:
            self.model = pickle.load(my_pickle)
        # loading tfidf csv file from AWS S3


        # loading stopwords

        print('Loading complete')
        return self.model, self.sw, self.tfidf


    def run(self):
        """
        run the detector, output the prediction
        """
        self.article = input("Enter your news article: ")
        self.corpus = self.get_corpus(self.article, self.sw)
        vector = self.model.transform(self.corpus)
        pred = self.model.predict(vector)
        if pred == 1:
            output = 'fake'
        else:
            output = 'not fake'

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

if __name__ == '__main__':

    detector = Detector()
    detector.run()


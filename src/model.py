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

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.metrics import AUC, BinaryAccuracy, Recall, Precision
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Testing and optimization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.regression import mean_absolute_error, mean_squared_error, r2_score

# import module
from src.pipeline import *

# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer
import pickle
import datetime as dt


class Model(object):
    """
    A class of model objects
    """

    def __init__(self):
        """takes a list of selected models, X, y and k for kfold, train each model and compare the model metrics """
        self.X = None
        self.X_pad = None
        self.X_lstm = None
        self.X_mlp = None
        self.y = None
        self.df = None
        self.baseline_prob = None
        self.metrics = ['AUC', 'BinaryAccuracy', 'Recall', 'Precision']

    def load_pickle(self, bucket, key):
        """
        load pickled df data from s3
        """
        s3 = boto3.resource('s3')
        bucket = bucket
        key = key
        self.df = pickle.loads(s3.Bucket(bucket).Object(key).get()['Body'].read())
        return None

    def save_pickle(self, bucket, file):
        """
        save pickle and upload to S3
        :return: None
        """
        s3 = boto3.resource('s3')
        data = open(file, "rb")
        key = bucket + '/' + file
        s3.Bucket(bucket).put_object(Key=key, Body=data)
        return self

    def savemodel(self,model,name):
        """
        model: trained keras model
        name:
        :return: None
        """
        path = 's3://fakenewscorpus/savedmodel/{}'.format(name)
        model.save(path)
        return None

class Baseline(Model):
    """
    A child class of baseline model that simply predict outcome based on probablity
    """

    def __init__(self):
        super().__init__()
        self.prob = []

    def fit(self):
        """
        fit the baseline model
        """
        n_pos, n_neg = balance(self.df)
        threshold = n_pos/(n_pos + n_neg)
        for row in self.X:
            self.prob.append(np.random.uniform)
        return None

    def predict(self):
        """
        predict base on probability
        """
        y_pred = []
        if self.prob > self.threshold:
            y_pred.append(0)
        else:
            e_pred.append(1)
        return y_pred


class RNN(Model):
    """
    A child class of lstm model
    """
    def __init__(self):
        super().__init__()
        self.sw = None
        self.model = None

    def text_prep(self, df, max_feature=10000, maxlen=10000, randseed=1):
        """
        Preprocess text for LSTM model
        :return:
        """
        # text_preprocess for LSTM model
        self.df = clean(self.df)
        self.X_lstm = self.df['content']
        self.y = self.df['label']
        self.t = Tokenizer(num_words=max_feature)
        self.t.fit_on_texts(self.df['content'])
        self.X = self.t.texts_to_sequences(self.df['content'])
        self.X_pad = pad_sequences(self.X, maxlen=maxlen)
        X_train, X_test, y_train, y_test = train_test_split(X_pad, y, random_state=randseed)
        self.X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        self.X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        return None

    def fit(self, num_words=5000, maxlen=1000):
        """
        return: LSTM model
        """
        self.model = Sequential()

        self.model.add(LSTM(units=128, return_sequences=True, input_shape=(1, 500)))
        self.model.add(LSTM(128), )
        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=self.metrics)

        # print(model3.summary())

        self.model.fit(X_train[:-10000], y_train[:-10000], epochs=43, batch_size=128, verbose=1, validation_data=(X_train[-10000:], y_train[-10000:]))

        self.savemodel(self.model, 'LSTM_model')
        return None

    def predict(self, X_test):
        """
        :param X_test: matrix
        :return: y_pred
        """
        y_pred = self.model.predict(X_test, batch_size=128, verbose=1, use_multiprocessing=True)
        return y_pred

    def score(self, X_test, y_test):
        """
        :param y_pred:
        :param y_test:
        :return: model metrics
        """
        self.scores = self.model.evaluate(X_test, y_test)
        for i, metric in enumerate(self.metrics):
            print('{} is {}.'.format(metric, self.scores[i]))


class MLP(Model):
    """
    A child class of mlp model
    """
    def __init__(self):
        super().__init__()
        self.sw = None
        self.model = None
        self.tfidf = None

    def text_prep(self, max_feature=10000, randseed=1, ngram=1):
        """
        :param df: pandas DataFrame
        :param max_feature:
        :param randseed:
        :param ngram:
        :return: None
        """
        # text preprocess for MLP model
        self.sw = pd.read_csv('./data/sw1k.csv')['term'].to_numpy()
        self.df['token'] = tokenize(self.df['content'], self.sw)
        self.X = self.df['tokens']
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=randseed)
        self.bow, tf, self.tfidf, cv, self.tv = vectorize(X_train, max_features=max_feature, ngram=ngram)

        return None

    def fit(self):
        """
        Input: tfidf - np array, y_train
        Return: trained mlp model
        """
        input_dim = self.tfidf.shape[1]

        self.model = Sequential()

        self.model.add(Dense(units=500, activation='relu', input_dim=input_dim))
        self.model.add(Dense(units=1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=self.metrics)

        self.model.fit(self.tfidf[:-10000], y_train[:-10000],
                   epochs=30, batch_size=batch_size, verbose=1,
                   validation_data=(tfidf[-10000:], y_train[-10000:]))

        self.savemodel(self.model, 'MLP_model')
        return self

    def predict(self, X_test):
        """
        :param X_test: matrix
        :return: y_pred
        """
        y_pred = self.model.predict(X_test, batch_size=128, verbose=1, use_multiprocessing=True)
        return y_pred

    def score(self, X_test, y_test):
        """
        :param y_pred:
        :param y_test:
        :return: model metrics
        """
        self.scores = self.model.evaluate(X_test, y_test)
        for i, metric in enumerate(self.metrics):
            print('{} is {}.'.format(metric, self.scores[i]))


# if __name__ == '__main__':















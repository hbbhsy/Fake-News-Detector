# Classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Testing and optimization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.regression import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import numpy as np
import sys
import os.path
from collections import defaultdict


class Model(object):
    """
    A modeling class that
    """

    def __init__(self, models, X, y, k=5):
        """takes a list of selected models, X, y and k for kfold, train each model and compare the model metrics """
        self.X = X
        self.y = y
        self.models = models
        self.k = k

    def train(self, model, X_train, y_train):

        m = model
        model = m.fit(X_train, y_train)

        return model

    def model_metrics(self, trained_model, X_test, y_test):

        y_hat = trained_model.predict(X_test)
        score = trained_model.score(X_test, y_test)
        f1 = f1_score(y_test, y_hat)
        precision = precision_score(y_test, y_hat)
        recall = recall_score(y_test, y_hat)
        roc_auc = roc_auc_score(y_test, y_hat)

        return score, f1, precision, recall, roc_auc

    def compare_models(self, models):
        """
        comparing models based on cross-validation , f1 score
        """
        kf = KFold(n_splits=self.k)

        for model in models:

            for train_ind, test_ind in kf.split(self.X):
                X_train, X_test = self.X[train_ind], self.X[test_ind]
                y_train, y_test = self.y[train_ind], self.y[test_ind]

                m = self.train(model, X_train, y_train)
                m_score = (self.model_metrics(m, X_test, y_test))

                # print ('{} metrics: {}'.format(model, m_scores))













from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model



# import module
from pipeline import *
from model import *

# import libraries
import boto3, re, sys, math, json, os, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def demo_detector(article, tv, model):
    demo = tv.transform([article])
    demo = demo.todense()
    print('\nThinking......\n\n')
    if model.predict(demo) <= 0.5:

        print('This article is non-fake.\n')
    else:
        print('This article is fake.\n')

if __name__ == '__main__':
    demo_model = load_model('../saved_model/demo_model')
    demo_tv = pickle.load(open("demo_tv.pickle", "rb"))
    X_demo = input("Enter the news you want to test: ")
    demo_detector(X_demo, demo_tv, demo_model)


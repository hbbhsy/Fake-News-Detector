import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import sys
import unicodedata


def clean(df, labeling=True):
    """
    Input: df: pandas DataFrame

   Remove unused features, relabeling

   Return: df: pandas DataFrame
   """
    df = df.drop(
        ['id', 'Unnamed: 0', 'domain', 'url', 'scraped_at', 'inserted_at', 'updated_at', 'keywords', 'meta_description',
         'meta_keywords', 'tags', 'summary', 'source'], axis=1)
    df = df[df['type'] != 'unknown']
    df = df.dropna()
    if labeling:
        df['label'] = df['type'].replace({'rumor': 1, 'hate': 0, 'unreliable': 1, 'conspiracy': 0, 'clickbait': 1, 'satire': 0,
                                          'fake': 1, 'reliable': 0, 'bias': 0, 'political': 0, 'junksci': 0})
    # df = df.reset_index()
    return df


def remove_accents(input_str):
    """
    remove accents for an input_str
    return str
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii.decode()


def cleanText(wordSeries):
    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)).startswith('P'))

    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        return only_ascii.decode()

    def remove_punctuation(text):
        return text.translate(tbl)

    # remove punctuation
    wordSeries = wordSeries.apply(lambda x: remove_punctuation(x))
    wordSeries = wordSeries.apply(lambda x: ''.join([i for i in x if not i.isdigit()]))  #remove digits
    wordSeries = wordSeries.apply(lambda x: x.lower())#lower cases
    wordSeries = wordSeries.apply(lambda x: x.replace('<br >', ' '))#remove html
    wordSeries = wordSeries.apply(lambda x: x.replace('<br>', ' '))#remove html
    wordSeries = wordSeries.apply(lambda x: x.replace('\n', ' '))#remove html
    wordSeries = wordSeries.apply(lambda x: x.replace('\n\n', ' '))
    wordSeries = wordSeries.apply(lambda x: x.replace('$', ' '))
    wordSeries = wordSeries.apply(lambda x: x.replace('>', ' '))
    wordSeries = wordSeries.apply(lambda x: remove_accents(x))
    wordSeries = wordSeries.apply(lambda x: x.replace('`', ''))#remove extra punctuation
    # wordSeries = wordSeries.apply(lambda x: x.replace(' id ', ' '))
    return wordSeries


def tokenize(documents, stopwords):
    """
    Input: Array of documents

    Remove stopwords, html, punctuation, digits
    Tokenize each document
    Lemmatize tokens for each document

    Return: Array of tokenized documents
    """
    documents = cleanText(documents)
    docs = [word_tokenize(content) for content in documents] #tockenize row by row
    # stopwords_=set(stopwords.words('english'))
    # stopwords_=pd.read_csv('../data/sw1k.csv')['term'].to_numpy()
    punctuation_ = set(string.punctuation)
    def filter_tokens(sent):
        return([w for w in sent if not w in stopwords and not w in punctuation_]) #remove stopword
    docs = list(map(filter_tokens,docs))
    lemmatizer = WordNetLemmatizer()
    docs_lemma = [[lemmatizer.lemmatize(word) for word in words] for words in docs]
    return docs_lemma


def vectorize(documents, max_features=50000, ngram=2):
    """
    Input: tokenized documents

    Compute Bag-of-Word, TF, TFIDF using sklearn

    Return: bow,tf,tfidf
    """
    corpus = [' '.join(row) for row in documents]
    # stopwords = pd.read_csv('../data/sw1k.csv')['term'].to_numpy()
    cv = CountVectorizer(max_features=max_features, ngram_range=(1, ngram))
    tf = cv.fit_transform(corpus).todense()
    tv = TfidfVectorizer(max_features=max_features)
    tfidf = tv.fit_transform(corpus).todense()
    bow = tv.get_feature_names()

    return bow, tf, tfidf, cv, tv

def balance(df):
    n_pos = len(df[df['label']==1])
    n_neg = len(df[df['label']==0])
    return n_pos, n_neg


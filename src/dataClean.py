import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string
import sys
import unicodedata


def tokenize(documents):

    def cleanText(wordSeries):
        tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))

        def remove_punctuation(text):
            return text.translate(tbl)

        wordSeries = wordSeries.apply(lambda x: remove_punctuation(x))
        wordSeries = wordSeries.apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
        wordSeries = wordSeries.apply(lambda x: x.lower())
        wordSeries = wordSeries.apply(lambda x: x.replace('<br >', ' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('<br>', ' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('\n', ' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('\n\n', ' '))
        wordSeries = wordSeries.apply(lambda x: x.replace('`', ''))
        wordSeries = wordSeries.apply(lambda x: x.replace(' id ', ' '))
        return wordSeries

    documents = cleanText(documents)
    docs = [word_tokenize(content) for content in documents]
    stopwords_=set(stopwords.words('english'))
    def filter_tokens(sent):
        return([w for w in sent if not w in stopwords_])
    docs=list(map(filter_tokens,docs))
    lemmatizer = WordNetLemmatizer()
    docs_lemma = [[lemmatizer.lemmatize(word) for word in words] for words in docs]
    return docs_lemma
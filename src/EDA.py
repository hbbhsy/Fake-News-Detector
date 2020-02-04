import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import operator
import string
import sys
import unicodedata

def getWordForTypes(df):

    def createDict(token):
        words = {}
        for row in token:
            for word in row:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        sorted_d = dict(sorted(words.items(), key=operator.itemgetter(1), reverse=True))
        return sorted_d

    types = df['type'].unique()
    type_df=[]
    for i in types:
        type_df.append(df[df['type']==i])
    # fake = df[df['type'] == 'fake']
    # hate = df[df['type'] == 'hate']
    # rumor = df[df['type'] == 'rumor']
    # satire = df[df['type'] == 'satire']
    # political = df[df['type'] == 'political']
    # clickbait = df[df['type'] == 'clickbait']
    # conspiracy = df[df['type'] == 'conspiracy']
    # unreliable = df[df['type'] == 'unreliable']
    # junksci = df[df['type'] == 'junksci']
    # bias = df[df['type'] == 'bias']

    type_word_lst={}
    for i,t in enumerate(types):
        type_word_lst[t]=createDict(type_df[i]['tokens'])
    return type_word_lst

# def topNWords(dict_of_words, N):
#     top_n=max(dict_of_words,key=dict_of_words.get)[:N]
#     return top_n


from collections import defaultdict
# dct = defaultdict(list)

def top_n(d, n):
    dct = defaultdict(list)
    for k, v in d.items():
        dct[v].append(k)
    return sorted(dct.items())[-n:][::-1]

def plotWordCould(ax,bow)：
    ax.figure(figsize = (20,10))
    wc = WordCloud(background_color="white",width=1000,height=1000, max_words=100,
               relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(bow)
plt.grid(False)
plt.title("Fake News Words",size=36)
plt.axis('off')
plt.imshow(wc)
plt.savefig('EDA/fake_words.png')
plt.show()




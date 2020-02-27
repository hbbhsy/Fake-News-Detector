import pandas as pd
import operator
from src.pipeline import tokenize
from src.pipeline import clean
from src.pipeline import vectorize
from collections import defaultdict
from sklearn.decomposition import NMF


class EDA(object):
    """
    A EDA Class
    """

    def __init__(self):
        self.df = None
        self.types = None
        self.bow = None
        self.tf = None
        self.tfidf = None
        self.type_word_lst = {}
        self.W = None
        self.H = None
        self.H_df = None
        self.cv = None
        self.tv = None
        self.sw = None

    def process(self, df, sw):
        """
        take a list of stopwords
        remove extra columns of self.df, tokenized self.df
        """
        self.df = clean(df)
        self.sw = sw
        documents = tokenize(self.df['content'], self.sw)
        self.df['tokens'] = documents
        self.types = self.df['type'].unique()
        return self

    def wordCounter(self, tokens):
        """Take array of tokenized documents
        Get sorted word counts dict
        """
        words = {}
        for row in tokens:
            for word in row:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

        sorted_d = dict(sorted(words.items(),
                               key=operator.itemgetter(1),
                               reverse=True))
        return sorted_d

    def getWordsForTypes(self):
        """
        get words for each type
        """
        type_df = []
        for i in self.types:
            type_df.append(self.df[self.df['type'] == i])

        for i, t in enumerate(self.types):
            self.type_word_lst[t] = self.wordCounter(type_df[i]['tokens'])
        return self.type_word_lst

    def top_words(self, n):
        """
        return top n words for a dict representing  news type
        """
        def top_n(d, n):
            """
            helper function to return top n keys by values in given dict
            """
            dct = defaultdict(list)
            for k, v in self.type_word_lst[t].items():
                dct[v].append(k)
            return sorted(dct.items())[-n:][::-1]

        for t in self.types:
            t_n = []
            for i in top_n(self.type_word_lst[t], n):
                t_n.append(i[1][0])
            print('{}: {}\n'.format(t, t_n))

    def get_vector(self):
        self.bow, self.tf, self.tfidf, self.cv, self.tv = vectorize(self.df['tokens'])
        return self.bow, self.tf, self.tfidf

    def doNMF(self, n):
        """
        Call sklearn NMF to perform basic NMF,
        return W, H for self.df
        """
        nmf = NMF(n_components=n)
        nmf.fit(self.tf)
        self.W = nmf.transform(self.tf)
        self.H = nmf.components_
        self.H_df = pd.DataFrame(self.H, columns=self.bow)
        return self





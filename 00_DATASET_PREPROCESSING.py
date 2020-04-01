# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys
sys.path

sys.path.insert(0, 'classes_functions')

import numpy as np
import text_normalizer as tn
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

data_df = pd.read_csv('data/IMDB Dataset.csv')
# -

data_df

# +
# how many empty documents are there?

total_nulls = data_df[data_df['review'].str.strip() == ""].shape[0]
print("Empty documents:", total_nulls)

# +
import nltk
stopword_list = nltk.corpus.stopwords.words('english')
# just to keep negation if any in bi-grams
stopword_list.remove('no')
stopword_list.remove('not')

norm_corpus = tn.normalize_corpus(corpus=data_df['review'], html_stripping=True,
                                 contraction_expansion=True, accented_char_removal=True,
                                 text_lower_case=True, text_lemmatization=True,
                                 text_stemming=False, special_char_removal=True,
                                 remove_digits=True, stopword_removal=True,
                                 stopwords=stopword_list)
data_df['clean review'] = norm_corpus
# -

data_df.head()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data_df['sentiment'])

le.classes_

pd.Series(le.transform(data_df['sentiment']))

data_df['sentiment label'] = pd.Series(le.transform(data_df['sentiment']))

data_df

data_df.to_csv('clean_IMDB Dataset.csv', index=False)

data_df['sentiment'].value_counts()



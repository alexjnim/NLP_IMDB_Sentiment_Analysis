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

data_df = pd.read_csv('clean_IMDB Dataset.csv')
# -

data_df.head()

# +
from sklearn.model_selection import train_test_split

train_corpus, test_corpus, train_label_nums, test_label_nums, train_label_names, test_label_names = train_test_split(
    np.array(data_df['clean review']),
                                         np.array(data_df['sentiment label']),
                                         np.array(data_df['sentiment']),
                                         test_size=0.33, random_state=42)
train_corpus.shape, test_corpus.shape
# -

from collections import Counter
trd = dict(Counter(train_label_names))
tsd = dict(Counter(test_label_names))
(pd.DataFrame([[key, trd[key], tsd[key]] for key in trd],
             columns=['sentiment', 'Train Count', 'Test Count'])
.sort_values(by=['Train Count', 'Test Count'],
             ascending=False))

# ### TD_IDF Model

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., norm="l2",
                     use_idf=True, smooth_idf=True)

tv_train_features = tv.fit_transform(train_corpus)
tv_test_features = tv.transform(test_corpus)


print('TF-IDF model:> Train features shape:', tv_train_features.shape,
      ' Test features shape:', tv_test_features.shape)
# -

tv_matrix = tv_train_features.toarray()
vocab = tv.get_feature_names()
pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)

vocab[5000:5500]

# ### ML algorithms on TF-IDF model

import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# +
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

mnb = MultinomialNB(alpha=1)
lr = LogisticRegression(penalty='l2', max_iter=100, C=1, random_state=42)
svm = LinearSVC(penalty='l2', C=1, random_state=42)
svm_sgd = SGDClassifier(loss='hinge', penalty="l2", max_iter=5, random_state=42)
rfc = RandomForestClassifier(n_estimators=10, random_state=42)

scores_df = pd.DataFrame(columns = ['train_accuracy', 'test_accuracy', 'fit_time'])

models = [mnb, lr, svm, svm_sgd, rfc]
names = ['Multinomial Naive Bayes', 'Logistic Regression', 'Linear SVC',
         'SGD Classifier', 'Random Forest Classifier']

for model, name in zip(models, names):
    temp_list = []
    print(name)

    model.fit(tv_train_features, train_label_names)
    scores = cross_validate(model, tv_train_features, train_label_names,
                            scoring=('accuracy'),
                            return_train_score=True, cv=10)

    for score in ['accuracy']:
        mean_score = scores['train_score'].mean()
        print('train {} mean : {}'.format(score, mean_score))
        temp_list.append(mean_score)

    test_score = model.score(tv_test_features, test_label_names)
    temp_list.append(test_score)
    print('test accuracy mean: {}'.format(test_score))

    temp_list.append(scores['fit_time'].mean())
    print('average fit time: {} \n'.format(scores['fit_time'].mean()))
    scores_df.loc[name] = temp_list

# -

scores_df

# ### try TD-IDF with n-grams

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., norm="l2",
                     use_idf=True, smooth_idf=True, ngram_range = (1,2))

tv_train_features = tv.fit_transform(train_corpus)
tv_test_features = tv.transform(test_corpus)


print('TF-IDF model:> Train features shape:', tv_train_features.shape,
      ' Test features shape:', tv_test_features.shape)
# -

tv_matrix = tv_train_features.toarray()
vocab = tv.get_feature_names()
pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)

vocab[50000:55000]

for model, name in zip(models, names):
    temp_list = []
    print(name)

    model.fit(tv_train_features, train_label_names)
    scores = cross_validate(model, tv_train_features, train_label_names,
                            scoring=('accuracy'),
                            return_train_score=True, cv=10)

    for score in ['accuracy']:
        mean_score = scores['train_score'].mean()
        print('train {} mean : {}'.format(score, mean_score))
        temp_list.append(mean_score)

    test_score = model.score(tv_test_features, test_label_names)
    temp_list.append(test_score)
    print('test accuracy mean: {}'.format(test_score))

    temp_list.append(scores['fit_time'].mean())
    print('average fit time: {} \n'.format(scores['fit_time'].mean()))
    scores_df.loc[name] = temp_list


scores_df



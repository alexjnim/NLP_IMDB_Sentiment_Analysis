# -*- coding: utf-8 -*-
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

# let's remove null values
data_df = data_df.dropna().reset_index(drop=True)
data_df.info()

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
trd = dict(Counter(train_label_nums))
tsd = dict(Counter(test_label_nums))
(pd.DataFrame([[key, trd[key], tsd[key]] for key in trd],
             columns=['sentiment', 'Train Count', 'Test Count'])
.sort_values(by=['Train Count', 'Test Count'],
             ascending=False))

# ### gridsearch pipeline with linear svc and tf-idf

# +
# Tuning our Multinomial Naïve Bayes model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

svm_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                        ('svm', LinearSVC())
                       ])

### here we evaluate this on bigrams and unigrams tf-idf and change the alpha value of MNB

param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)],
              'svm__C': [0.5, 1.0, 1.5],
              'svm__penalty': ['l1', 'l2']
                }

gs_svm = GridSearchCV(svm_pipeline, param_grid, cv=5, verbose=2)
gs_svm = gs_svm.fit(train_corpus, train_label_nums)
# -

gs_svm.best_estimator_.get_params()

cv_results = gs_svm.cv_results_
results_df = pd.DataFrame({'rank': cv_results['rank_test_score'],
                           'params': cv_results['params'],
                           'cv score (mean)': cv_results['mean_test_score'],
                           'cv score (std)': cv_results['std_test_score']}
              )
results_df = results_df.sort_values(by=['rank'], ascending=True)
pd.set_option('display.max_colwidth', 100)
results_df

best_svm_test_score = gs_svm.score(test_corpus, test_label_nums)
print('Test Accuracy :', best_svm_test_score)

# ## ELI5 evaluation

svm_best = gs_svm.best_estimator_

svm_best[1]

import eli5
# see top predictors in each class 
eli5.show_weights(svm_best[1], vec=svm_best[0], top=40)

test_corpus[8]

# test for a given comment's best features 
eli5.show_prediction(svm_best[1], test_corpus[8], vec=svm_best[0], top=10)

# ### model performance evaluation with Linear SVM

# +
import model_evaluation_utils as meu

svm_predictions = gs_svm.predict(test_corpus)
unique_classes = list(set(test_label_nums))
meu.get_metrics(true_labels=test_label_nums, predicted_labels=svm_predictions)
# -

meu.display_classification_report(true_labels=test_label_nums,
                                  predicted_labels=svm_predictions,
                                  classes=unique_classes)

from confusion_matrices import confusion_matrices

confusion_matrices(test_label_nums, svm_predictions)

# ### checking mismatched values

# Extract test document row numbers
train_idx, test_idx = train_test_split(np.array(range(len(data_df['review']))), test_size=0.33, random_state=42)
test_idx

svm_predictions = gs_svm.predict(test_corpus)
test_df = data_df.iloc[test_idx]
test_df['predicted label'] = svm_predictions
test_df.head()

pd.set_option('display.max_colwidth', 200)
res_df = (test_df[(test_df['sentiment label'] == 1)
                  & (test_df['predicted label'] == 0)])
res_df



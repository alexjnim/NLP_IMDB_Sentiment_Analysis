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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def confusion_matrices(y, y_pred):
    y_pred = y_pred.round()
    confusion_mat = confusion_matrix(y, y_pred)
    sns.set_style("white")
    plt.matshow(confusion_mat, cmap=plt.cm.gray)
    plt.show()
    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    normalised_confusion_mat = confusion_mat/row_sums
    print(confusion_mat, "\n")
    print(normalised_confusion_mat)
    plt.matshow(normalised_confusion_mat, cmap=plt.cm.gray)
    plt.show()
    print('the precision score is : ', precision_score(y, y_pred))
    print('the recall score is : ', recall_score(y, y_pred))
    print('the f1 score is : ', f1_score(y, y_pred))
    print('the accuracy score is : ', accuracy_score(y, y_pred))
    print('the roc_auc score is : ', roc_auc_score(y, y_pred))
    return
# -



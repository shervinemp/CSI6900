import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, precision_score, recall_score

from data_utils import CSVData
from rq3 import (COUNT, EXP_REPEAT, MAX_REPEAT, SEED, fit_cum_range, prep_data,
                 train)


def test(scores, desc, X, y, output_file='rq3.txt'):
    models = scores['estimator']
    preds = [model.predict(X) for model in models]
    precisions = [precision_score(y, pred) for pred in preds]
    recalls = [recall_score(y, pred) for pred in preds]
    f1s = [f1_score(y, pred) for pred in preds]
    with open(output_file, 'at') as f:
        f.write(f'Approach: {desc}\n')
        f.write(f'Inputs: {list(X.columns)}\n')
        f.write(f'Output: {list(y.columns)}\n')
        f.write(f'Precision scores: {scores["test_precision"]}\n')
        f.write(f'Mean precision score: {scores["test_precision"].mean()}\n')
        f.write(f'Recall scores: {scores["test_recall"]}\n')
        f.write(f'Mean recall score: {scores["test_recall"].mean()}\n')
        f.write(f'F1 scores: {scores["test_f1"]}\n')
        f.write(f'Mean F1 score: {scores["test_f1"].mean()}\n')
        f.write(f'Test precision: {precisions}\n')
        f.write(f'Test mean precision: {np.mean(precisions)}\n')
        f.write(f'Test best precision: {np.max(precisions)}\n')
        f.write(f'Test recall: {recalls}\n')
        f.write(f'Test mean recall: {np.mean(recalls)}\n')
        f.write(f'Test best recall: {np.max(recalls)}\n')
        f.write(f'Test F1: {f1s}\n')
        f.write(f'Test mean F1: {np.mean(f1s)}\n')
        f.write(f'Test best F1: {np.max(f1s)}\n')
        f.write('\n')

if __name__  == '__main__':
    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1])
    df_train, df_test = data.get(min_rep=EXP_REPEAT, max_rep=EXP_REPEAT,
                                 count=COUNT, split=0.8, random_state=SEED)
    X_train, y_train, sl_train, hl_train = prep_data(df_train)
    X_test, y_test, sl_test, hl_test = prep_data(df_test)

    methods = ('dt', 'rf', 'svm', 'mlp')
    mparams = ({'max_depth': 5},
               {'max_depth': 5},
               {},
               {'max_iter': 1000})

    if os.path.exists('rq3.txt'):
        os.remove('rq3.txt')
    for method, kwparams in zip(methods, mparams):
        for X_train_, X_test_ in zip(fit_cum_range(X_train, MAX_REPEAT),
                                     fit_cum_range(X_test, MAX_REPEAT)):
            m, d = train(X_train_, sl_train, method=method, **kwparams)
            test(m, d, X_test_, sl_test, output_file='rq3.txt')

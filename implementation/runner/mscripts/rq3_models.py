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
    with open(output_file, 'a') as f:
        f.write(f'Approach: {desc}\n')
        f.write(f'Inputs: {list(X.columns)}\n')
        f.write(f'Output: {list(y.columns)}\n')
        f.write(f'Precision scores: {scores["test_precision"]}\n')
        f.write(f'Mean precision score: {scores["test_precision"].mean()}\n')
        f.write(f'Recall scores: {scores["test_recall"]}\n')
        f.write(f'Mean recall score: {scores["test_recall"].mean()}\n')
        f.write(f'F1 scores: {scores["test_f1"]}\n')
        f.write(f'Mean F1 score: {scores["test_f1"].mean()}\n')
        f.write(f'Test set precision: {precisions}\n')
        f.write(f'Test set mean precision: {np.mean(precisions)}\n')
        f.write(f'Test set best precision: {np.max(precisions)}\n')
        f.write(f'Test set recall: {recalls}\n')
        f.write(f'Test set mean recall: {np.mean(recalls)}\n')
        f.write(f'Test set best recall: {np.max(recalls)}\n')
        f.write(f'Test set F1: {f1s}\n')
        f.write(f'Test set mean F1: {np.mean(f1s)}\n')
        f.write(f'Test set best F1: {np.max(f1s)}\n')

if __name__  == '__main__':
    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1])
    df_train, df_test = data.get(min_rep=EXP_REPEAT, max_rep=EXP_REPEAT,
                                 count=COUNT, split=0.8, random_state=SEED)
    X_train, y_train, sl_train, hl_train = prep_data(df_train)
    X_test, y_test, sl_test, hl_test = prep_data(df_test)

    os.remove('rq3.txt') if os.path.exists('rq3.txt') else None
    for method in ('dt', 'svm', 'mlp'):
        for X_train_, X_test_ in zip(fit_cum_range(X_train, MAX_REPEAT),
                                     fit_cum_range(X_test, MAX_REPEAT)):
            m, d = train(X_train_, sl_train, method=method, max_depth=5)
            test(m, d, X_test_, sl_test, output_file='rq3.txt')

from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from rq3_dataset import balance
import logging as log
from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import os

log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def train(df_main, cv=5, approach='dt', output_file='rq3.txt', **kwargs):
    flen = kwargs.get('fitnesses_len', 0)
    df_train = df_main.sample(frac=0.8, random_state=200)
    df_test = df_main.drop(df_train.index).reset_index(drop=True)
    df_train = balance(df_train, output_file=None)
    train = df_train.copy()
    test = df_test.copy()
    with open(output_file, 'a') as f:
        if approach == 'dt':
            scores, desc = trainDecisionTree(train, cv=cv, **kwargs)
        elif approach == 'svm':
            scores, desc = trainSVM(train, cv=cv, **kwargs)
        elif approach == 'mlp':
            scores, desc = trainMLP(train, cv=cv, **kwargs)
        models = scores['estimator']
        predictions = [model.predict(test.iloc[:, :-1].values) for model in models]
        precisions = [precision_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        recalls = [recall_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        f1s = [f1_score(test.iloc[:, -1].values, prediction) for prediction in predictions]

        f.write('*' * 25 + '  With all the features  ' + '*' * 25 + '\n')
        f.write(f'Approach: {desc}\n')
        f.write(f'Inputs: {list(df_main.columns[:-1])}\n')
        f.write(f'Output: {list(df_main.columns[-1])}\n')
        f.write(f'Number of fitnesses included in input: {flen}\n')
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
    df_main = pd.read_csv('dataset.csv')
    df_main = df_main.sample(frac=1, random_state=42).reset_index(drop=True)
    df_main = df_main.drop(['Y0', 'Y1', 'Y2', 'Y3'], axis=1)
    
    ##  Visualize number of classes...
    # sns.set_theme()
    # plt.subplots(1, 5, figsize=(20, 5), layout='tight')
    # plt.subplot(1, 5, 1)
    # sns.countplot(x='Y0', data=df_main)
    # plt.xlabel('Fitness 1')
    # plt.subplot(1, 5, 2)
    # sns.countplot(x='Y1', data=df_main)
    # plt.xlabel('Fitness 2')
    # plt.subplot(1, 5, 3)
    # sns.countplot(x='Y2', data=df_main)
    # plt.xlabel('Fitness 3')
    # plt.subplot(1, 5, 4)
    # sns.countplot(x='Y3', data=df_main)
    # plt.xlabel('Fitness 4')
    # plt.subplot(1, 5, 5)
    # sns.countplot(x='Y', data=df_main)
    # plt.xlabel('Overall')
    # plt.show()

    os.remove('rq3.txt') if os.path.exists('rq3.txt') else None
    approaches = ['dt', 'svm', 'mlp']
    for approach in approaches:
        ##  Without fitnesses...
        df = df_main.drop(['F0_0', 'F0_1', 'F0_2', 'F1_0', 'F1_1', 'F1_2', 'F2_0', 'F2_1', 'F2_2', 'F3_0', 'F3_1', 'F3_2'], axis=1)
        train(df, cv=5, approach=approach, output_file='rq3.txt', max_depth=5, fitnesses_len=0)
        ##  With 1 set of fitnesses...
        df = df_main.drop(['F0_1', 'F0_2', 'F1_1', 'F1_2', 'F2_1', 'F2_2', 'F3_1', 'F3_2'], axis=1)
        train(df, cv=5, approach=approach, output_file='rq3.txt', max_depth=5, fitnesses_len=1)
        ##  With 2 sets of fitnesses...
        df = df_main.drop(['F0_2', 'F1_2', 'F2_2', 'F3_2'], axis=1)
        train(df, cv=5, approach=approach, output_file='rq3.txt', max_depth=5, fitnesses_len=2)
        ##  With 3 sets of fitnesses...
        df = df_main.copy()
        train(df, cv=5, approach=approach, output_file='rq3.txt', max_depth=5, fitnesses_len=3)

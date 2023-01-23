import logging as log
import sys
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.oversampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import CSVData, enum_cols, fit_cols, fit_labels, in_cols

SEED = 0
EXP_REPEAT = 10

random_ = np.random.RandomState(seed=SEED)

def train_model(model, X, y, cv=5):
    cv = StratifiedKFold(n_splits=cv, random_state=random_)
    scores = cross_validate(model, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
    desc = str(model)
    return scores, desc

def trainDecisionTree(X, y, cv=5, **kwargs):
    kwargs.setdefault('max_depth', 5)
    model = DecisionTreeClassifier(**kwargs)
    return train_model(model, X, y, cv)

def trainSVM(X, y, cv=5, **kwargs):
    model = SVC(**kwargs)
    return train_model(model, X, y, cv)

def trainMLP(X, y, cv=5, **kwargs):
    kwargs.setdefault('hidden_layer_sizes', (50, 100))
    kwargs.setdefault('learning_rate', 'adaptive')
    model = MLPClassifier(**kwargs)
    return train_model(model, X, y, cv)

def fit_range(X, end):
    for i in range(end):
        yield X.drop([f"{f}_{i}" for f, i in product(fit_cols, range(end, EXP_REPEAT))], axis=1) \
               .sample(frac=1, random_state=random_)

def trainModels(X, y, class_labels, cv=5, **kwargs):
    models = []
    for X_ in fit_range(X, 4):
        df_train, _ = balance(X_, y, class_labels, output_file=None)
        scores, desc = trainDecisionTree(df_train, class_labels, cv=cv, **kwargs)
        f1s = scores["test_f1"]
        models.append(scores['estimator'][np.argmax(f1s)])
    return models

def smartRandomSearch(X, models=None, method='or'):
    l_end = 4
    fit_iter_cols = [f"{f}_{i}" for f, i in product(fit_cols, range(l_end))]
    X_ = X[fit_iter_cols]
    if models is None:
        if method == 'and':
            w = np.logspace(1, l_end, num=l_end, base=1/2)
        elif method == 'or':
            w = 1 - np.logspace(1, l_end, num=l_end, base=1/2)
        w = w[np.newaxis, :].repeat(len(X_), axis=0)
    else:
        pred = np.array([m.predict(x) >= 0.5 for m, x in zip(models, fit_range(X_, l_end))]).T
        pred_df = pd.DataFrame(pred, columns=range(l_end))
        if method == 'and':
            w = pred_df.cumprod(axis=1)
        elif method == 'or':
            w = (pred_df.cumsum(axis=1) == 1).loc[:, ::-1].cumsum(axis=1).loc[:, ::-1]
    
    t = []
    for i in range(l_end):
        value_vars = fit_iter_cols[slice(i, len(fit_iter_cols), l_end)]
        var_name = f"{i}_fit"
        X_melt = X_.melt(value_vars=value_vars, var_name=var_name,
                            value_name=i, ignore_index=True)
        t.append(X_melt)
    df = pd.concat(t, axis=1)
    df[range(i)] = df[range(i)]* w
    df['min'] = df[range(i)].min(axis=1)
    df['mean'] = df[range(i)].mean(axis=1)
    df['f'] = df['0_fit'].apply(lambda x: x[:-2])
    df = pd.pivot(df, index='index', columns='f', values=['min', 'mean'])
    cnt = (w * range(l_end)).mean(axis=1)
    
    return df, cnt

def smartRandomSearchOneRun(scenarios, f_arr_detailed, models=None, method='or'):
    f_min, f_mean, cnt = smartRandomSearchOneIteration(scenarios[0], f_arr_detailed[0], models, method)
    f_mins = [f_min]
    f_means = [f_mean]
    cnts = [cnt]
    for i in range(1, len(scenarios)):
        f_min, f_mean, cnt = smartRandomSearchOneIteration(scenarios[i], f_arr_detailed[i], models, method)
        f_mins.append([min(f_min[j], f_mins[-1][j]) for j in range(len(f_min))]) 
        f_means.append([min(f_mean[j], f_means[-1][j]) for j in range(len(f_mean))])
        cnts.append(cnt)
    f_mins = np.array(f_mins)
    f_means = np.array(f_means)
    return f_mins, f_means, sum(cnts)

def smartRandomSearch(scenarios, f_arr_detailed, runs=20, models=None, method='or'):
    n_iterations = len(f_arr_detailed) // runs
    f_mins = []
    f_means = []
    cnts = []
    for i in range(runs):
        f_mins_prime, f_means_prime, cnt = smartRandomSearchOneRun(scenarios[i*n_iterations:(i+1)*n_iterations], f_arr_detailed[i*n_iterations:(i+1)*n_iterations], models, method)
        f_mins.append(f_mins_prime)
        f_means.append(f_means_prime)
        cnts.append(cnt)
    f_mins = np.array(f_mins)
    f_means = np.array(f_means)
    return f_mins, f_means, sum(cnts)

def balance(X, y, class_labels=None, smote_instance=SMOTE()):
    if class_labels is None:
        class_labels = y
    X_cols = X.columns
    y_cols = y.columns
    df = pd.concat([X, y], axis=1)
    df_resampled, _ = smote_instance.fit_resample(df, class_labels)
    return df_resampled[X_cols], df_resampled[y_cols]

if __name__  == '__main__':
    ##  Training models...
    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1], min_run=EXP_REPEAT)

    df_ = data.df_
    df_ = df_.groupby(level=in_cols) \
           .sample(EXP_REPEAT, random_state=random_) \
           .drop(columns=df_.columns.difference(fit_cols))
    
    max_delta = df_.max() - df_.min()
    delta = df_.groupby(level=in_cols) \
               .agg(lambda f: f.max() - f.min())
    smote_labels = (delta / max_delta >= 0.01)
    
    df_fit = df_.copy()
    df_fit['i'] = df_fit.groupby(level=in_cols).cumcount()
    df_fit = df_fit.pivot_table(index=in_cols, columns='i', values=fit_cols).set_index(in_cols)
    df_fit.columns = list(map(lambda x: '_'.join(map(str, x)), df_fit.columns))

    df = df_[in_cols].join(df_fit) \
                     .join(df_.agg(lambda f: (f > 0).any() & (f <= 0).any())) \
                     .reset_index()
    one_hot = pd.get_dummies(df[enum_cols])
    df = df.drop(columns=enum_cols).join(one_hot)

    y = df[fit_cols]
    X = df[df.columns.difference(fit_cols)]

    models = trainModels(X, y)
    
    ##  Smart random search random mode with or ...
    f_min_smart_random_or, f_mean_smart_random_or, cnt_random_or = smartRandomSearch(scenarios, f_arr_detailed, models=None, method='or')

    ##  Smart random search with models with or...
    f_min_smart_or, f_mean_smart_or, cnt_or = smartRandomSearch(scenarios, f_arr_detailed, models=models, method='or')

    ##  Smart random search random mode with and ...
    f_min_smart_random_and, f_mean_smart_random_and, cnt_random_and = smartRandomSearch(scenarios, f_arr_detailed, models=None, method='and')

    ##  Smart random search with models with and...
    f_min_smart_and, f_mean_smart_and, cnt_and = smartRandomSearch(scenarios, f_arr_detailed, models=models, method='and')

    ##  Random search for 10 repetitions...
    scenarios, f_arr10 = utils.minAndMeanFitnesses(result_detailed, normalize=False, n_reps=10)
    f_min10, f_mean10 = rq1.randomSearch(f_arr10)
    min_ = np.min(f_arr10, axis=(0, 1))
    max_ = np.max(f_arr10, axis=(0, 1))
    f_min10 = (f_min10 - min_) / (max_ - min_)
    f_mean10 = (f_mean10 - min_) / (max_ - min_)
    ##  Random search for 4 repetitions...
    scenarios, f_arr4 = utils.minAndMeanFitnesses(result_detailed, normalize=False, n_reps=4)
    f_min4, f_mean4 = rq1.randomSearch(f_arr4)
    f_min4 = (f_min4 - min_) / (max_ - min_)

    ##  Normalizing fitnesses for smart random search...
    f_min_smart_or = (f_min_smart_or - min_) / (max_ - min_)
    f_min_smart_random_or = (f_min_smart_random_or - min_) / (max_ - min_)
    f_min_smart_and = (f_min_smart_and - min_) / (max_ - min_)
    f_min_smart_random_and = (f_min_smart_random_and - min_) / (max_ - min_)


    # plotRS([f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or,f_min4, f_min10], f_mean10, ['RS-Smart-Random-AND', 'RS-Smart-models-AND', 'RS-Smart-Random-OR', 'RS-Smart-models-OR', 'RSw4REP', 'RSw10REP', 'RS'], show=True)
    plotBox([f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or,f_min4, f_min10], f_mean10, ['RS-Smart-Random-AND', 'RS-Smart-models-AND', 'RS-Smart-Random-OR', 'RS-Smart-models-OR', 'RSw4REP', 'RSw10REP', 'RS'], show=True)

    log.info(f'Number of iterations for smart RS in random mode OR: {cnt_random_or}')
    log.info(f'Number of iterations for smart RS with models OR: {cnt_or}')
    log.info(f'Number of iterations for smart RS in random mode AND: {cnt_random_and}')
    log.info(f'Number of iterations for smart RS with models AND: {cnt_and}')

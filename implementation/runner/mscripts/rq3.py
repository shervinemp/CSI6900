import logging as log
import sys
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import CSVData, enum_cols, fit_cols, fit_labels, in_cols

SEED = 0
EXP_REPEAT = 10
COUNT = 1000
MAX_REPEAT = 4

random_ = np.random.RandomState(seed=SEED)

def train_model(model, X, y, *, cv=5):
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_)
    scores = cross_validate(model, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
    desc = str(model)
    return scores, desc

def trainDecisionTree(X, y, *, cv=5, **kwargs):
    kwargs.setdefault('max_depth', 5)
    model = DecisionTreeClassifier(**kwargs)
    return train_model(model, X, y, cv=cv)

def trainSVM(X, y, *, cv=5, **kwargs):
    model = SVC(**kwargs)
    return train_model(model, X, y, cv=cv)

def trainMLP(X, y, *, cv=5, **kwargs):
    kwargs.setdefault('hidden_layer_sizes', (50, 100))
    kwargs.setdefault('learning_rate', 'adaptive')
    model = MLPClassifier(**kwargs)
    return train_model(model, X, y, cv=cv)

def fit_range(X, l_end=EXP_REPEAT):
    for l in range(l_end):
        dcols = [f"{f}_{i}" for f, i in product(fit_cols, range(l+1, EXP_REPEAT))]
        yield X.drop(X.columns.intersection(dcols), axis=1) \
               .sample(frac=1, random_state=random_)

def trainModels(X, y, class_labels=None, *, cv=5, **kwargs):
    if class_labels is None:
        class_labels = y
    models = []
    for X_ in fit_range(X, MAX_REPEAT):
        X_b, y_b = balance(X_, y, class_labels)
        scores, desc = trainDecisionTree(X_b, y_b, cv=cv, **kwargs)
        f1s = scores["test_f1"]
        models.append(scores['estimator'][np.argmax(f1s)])
    return models

def smartRandomSearch(X, models=None, method='or', l_end=MAX_REPEAT):
    if models is None:
        w = np.logspace(0, l_end-1, num=l_end, base=1/2)
        w = pd.DataFrame(w[np.newaxis, :].repeat(len(X), axis=0), columns=range(l_end))    
    else:
        pred = np.array([m.predict(x) >= 0.5 for m, x in zip(models, fit_range(X, l_end-1))]).T
        pred_df = pd.DataFrame(pred, columns=range(1, l_end))
        if method == 'and':
            w = pred_df.cumprod(axis=1)
        elif method == 'or':
            w = (~pred_df).cumprod(axis=1)
        w[0] = 1
        w.sort_index(axis=1, inplace=True)
    t = []
    for i in range(l_end):
        value_vars = [f"{f}_{i}" for f in fit_cols]
        var_name = f"{i}_fit"
        X_melt = X.melt(value_vars=value_vars, var_name=var_name,
                        value_name=i, ignore_index=False)
        t.append(X_melt)
    df = pd.concat(t, axis=1)
    n_cols = list(range(l_end))
    w_df = pd.concat([w / w.mean(axis=1).to_numpy()[:, np.newaxis]] * len(fit_cols), axis=0)
    df['min'] = (df[n_cols].cummin(axis=1) * w_df).mean(axis=1)
    df['mean'] = (df[n_cols] * w_df).mean(axis=1)

    df['f'] = df['0_fit'].apply(lambda x: x[:-2])
    df = pd.pivot(df, columns='f', values=['min', 'mean'])
    cnt = w_df.sum(axis=1)
    
    return df, cnt

def balance(X, y, class_labels=None, smote_instance=SMOTE(random_state=random_)):
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

    df_ = data._df[fit_cols].groupby(level=in_cols) \
                            .sample(EXP_REPEAT, random_state=SEED) \
                            .loc[random_.choice(data.indices, COUNT, replace=False)]
    
    max_delta = df_.max() - df_.min()
    delta = df_.groupby(level=in_cols) \
               .agg(lambda f: f.max() - f.min())
    labels = (delta / max_delta >= 0.01).any(axis=1) \
                                        .to_frame('label') \
                                        .astype(int) \
                                        .reset_index(drop=True)
    
    df_fit = df_.copy()
    df_fit['i'] = df_fit.groupby(level=in_cols).cumcount()
    df_fit = df_fit.pivot(columns=['i'], values=fit_cols)
    df_fit.columns = [f"{f}_{i}" for f, i in df_fit.columns]

    df = df_fit.join(df_.groupby(level=in_cols) 
                        .agg(lambda f: (f > 0).any() & (f <= 0).any())) \
               .reset_index()
    one_hot = pd.get_dummies(df[enum_cols])
    df = df.drop(columns=enum_cols).join(one_hot)

    y = df[fit_cols]
    X = df[df.columns.difference(fit_cols)]

    models = trainModels(X, labels)
    
    ##  Smart random search random mode with or ...
    df_random_or, cnt_random_or = smartRandomSearch(X, models=None, method='or')

    ##  Smart random search with models with or...
    df_smart_or, cnt_or = smartRandomSearch(X, models=models, method='or')

    ##  Smart random search random mode with and ...
    df_random_and, cnt_random_and = smartRandomSearch(X, models=None, method='and')

    ##  Smart random search with models with and...
    df_smart_and, cnt_and = smartRandomSearch(X, models=models, method='and')

    # ##  Random search for 10 repetitions...
    # scenarios, f_arr10 = utils.minAndMeanFitnesses(result_detailed, normalize=False, n_reps=10)
    # f_min10, f_mean10 = rq1.randomSearch(f_arr10)
    # min_ = np.min(f_arr10, axis=(0, 1))
    # max_ = np.max(f_arr10, axis=(0, 1))
    # f_min10 = (f_min10 - min_) / (max_ - min_)
    # f_mean10 = (f_mean10 - min_) / (max_ - min_)
    # ##  Random search for 4 repetitions...
    # scenarios, f_arr4 = utils.minAndMeanFitnesses(result_detailed, normalize=False, n_reps=4)
    # f_min4, f_mean4 = rq1.randomSearch(f_arr4)
    # f_min4 = (f_min4 - min_) / (max_ - min_)

    # ##  Normalizing fitnesses for smart random search...
    # f_min_smart_or = (f_min_smart_or - min_) / (max_ - min_)
    # f_min_smart_random_or = (f_min_smart_random_or - min_) / (max_ - min_)
    # f_min_smart_and = (f_min_smart_and - min_) / (max_ - min_)
    # f_min_smart_random_and = (f_min_smart_random_and - min_) / (max_ - min_)


    # # plotRS([f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or,f_min4, f_min10], f_mean10, ['RS-Smart-Random-AND', 'RS-Smart-models-AND', 'RS-Smart-Random-OR', 'RS-Smart-models-OR', 'RSw4REP', 'RSw10REP', 'RS'], show=True)
    # plotBox([f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or,f_min4, f_min10], f_mean10, ['RS-Smart-Random-AND', 'RS-Smart-models-AND', 'RS-Smart-Random-OR', 'RS-Smart-models-OR', 'RSw4REP', 'RSw10REP', 'RS'], show=True)

    # log.info(f'Number of iterations for smart RS in random mode OR: {cnt_random_or}')
    # log.info(f'Number of iterations for smart RS with models OR: {cnt_or}')
    # log.info(f'Number of iterations for smart RS in random mode AND: {cnt_random_and}')
    # log.info(f'Number of iterations for smart RS with models AND: {cnt_and}')

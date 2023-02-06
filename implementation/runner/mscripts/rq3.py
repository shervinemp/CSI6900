import re
import sys
import time
from typing import Sequence, Union

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_utils import CSVData, enum_cols, fit_cols, fit_labels, in_cols
from post_RS import RS
from utils import stat_test, static_vars, unstack_col_level

SEED = 0
EXP_REPEAT = 10
COUNT = 1000
ITER_COUNT = 50
MAX_REPEAT = 4

# Create a pseudorandom number generator with the specified seed
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

def trainRF(X, y, *, cv=5, **kwargs):
    kwargs.setdefault('max_depth', 5)
    model = RandomForestClassifier(**kwargs)
    return train_model(model, X, y, cv=cv)

def fit_cum_range(X, rang: Union[Sequence[int], int]):
    if type(rang) is int:
        rang = range(rang)
    yield from (fit_range(X, i+1) for i in rang)

@static_vars(regs=[re.compile(f"^{f}_\d+$") for f in fit_cols])
def get_fit_cols(X):
    return [col for col in X.columns if np.any([r.match(col) for r in get_fit_cols.regs])]

def fit_range(X, rang: Union[Sequence[int], int]):
    if type(rang) is int:
        rang = range(rang)
    dcols = list(filter(lambda f: int(f.split('_')[-1]) not in rang,
                        get_fit_cols(X)))
    return X.drop(dcols, axis=1)

def train_models(X, y, class_labels=None, *, cv=5, **kwargs):
    models = []
    for X_ in fit_cum_range(X, MAX_REPEAT):
        scores, desc = train(X_, y, class_labels, cv=cv, **kwargs)
        f1s = scores["test_f1"]
        models.append(scores['estimator'][np.argmax(f1s)])
    return models

def train(X, y, class_labels=None, method='dt', cv=5, **kwargs):
    if class_labels is None:
        class_labels = y
    X_b, y_b = balance_data(X, y, class_labels)
    y_b = y_b[y_b.columns[0]]
    if method == 'dt':
        scores, desc = trainDecisionTree(X_b, y_b, cv=cv, **kwargs)
    elif method == 'svm':
        scores, desc = trainSVM(X_b, y_b, cv=cv, **kwargs)
    elif method == 'mlp':
        scores, desc = trainMLP(X_b, y_b, cv=cv, **kwargs)
    elif method == 'rf':
        scores, desc = trainRF(X_b, y_b, cv=cv, **kwargs)
    else:
        raise ValueError(f"Method \"{method}\" not supported.")
    
    return scores, desc

def plotRS(df, show=True):
    output_file = 'rs_carla_iters.pdf'
    t0 = time.time()
    sns.set()
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(24, 5))
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        sns.lineplot(data=df, x='x', y=col, hue='method', ax=ax)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(label)
        ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    t1 = time.time()
    print(f'Plotting random search fitnesses took {t1-t0} seconds. Output file: {output_file}')
    plt.show() if show else None

def plotBox(df, output_file='rs_box.pdf', *, show=True):
    t0 = time.time()
    sns.set()
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(24, 5))
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        sns.boxplot(data=df, x='method', y=col, showmeans=True, ax=ax)
        ax.set_ylabel(label)
        ax.legend([], [], frameon=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    t1 = time.time()
    print(f'Plotting boxplots took {t1-t0} seconds. Output file: {output_file}')
    plt.show() if show else None

def smartFitness(X, models=None, method='or', max_rep=MAX_REPEAT, p_thresh=0.5):
    if method not in ('or', 'and', 'first'):
        raise ValueError(f"Method \"{method}\" is not valid.")
    if models is None:
        if method == 'and':
            visit_proba = np.logspace(1, max_rep, num=max_rep, base=p_thresh)
        elif method == 'or':
            visit_proba = np.logspace(1, max_rep, num=max_rep, base=1-p_thresh)
        elif method == 'first':
            visit_proba = np.ones(max_rep) * 0.5
        visit_proba = pd.DataFrame(visit_proba[np.newaxis, :].repeat(len(X), axis=0),
                                   columns=range(max_rep))    
    else:
        pred = np.array([m.predict(x) >= p_thresh \
                         for m, x in zip(models, fit_cum_range(X, max_rep))]).T
        pred_df = pd.DataFrame(pred, columns=range(max_rep))
        if method == 'and':
            visit_proba = pred_df.cumprod(axis=1)
        elif method == 'or':
            visit_proba = (~pred_df).cumprod(axis=1)
        elif method == 'first':
            visit_proba = pred_df[[0]].repeat(len(pred_df), axis=1)
    w = visit_proba.copy()
    w[-1] = 1
    w[max_rep] = 0
    w = w.sort_index(axis=1) \
         .diff(periods=-1, axis=1) \
         .drop(columns=[max_rep])
    
    t = []
    for i in range(max_rep):
        value_vars = [f"{f}_{i}" for f in fit_cols]
        var_name = f"{i}_fit"
        X_melt = X.melt(value_vars=value_vars, var_name=var_name,
                        value_name=i, ignore_index=False)
        t.append(X_melt)
    df = pd.concat(t, axis=1)

    w_df = pd.concat([w / w.mean(axis=1).to_numpy()[:, np.newaxis]] * len(fit_cols), axis=0)
    df_vals = df[range(max_rep)]
    df['min'] = (df_vals.cummin(axis=1) * w_df).mean(axis=1)
    df['mean'] = (df_vals.cumsum(axis=1) / range(1, df_vals.shape[1] + 1) * w_df).mean(axis=1)

    df['f'] = df['0_fit'].apply(lambda x: x[:-2])
    df = pd.pivot(df, columns='f', values=['min', 'mean'])
    cnt = visit_proba.sum(axis=1).sum()
    
    return df, cnt

def balance_data(X, y, class_labels=None, smote_instance=SMOTE(random_state=random_)):
    if class_labels is None:
        class_labels = y
    X_cols = X.columns
    y_cols = y.columns
    df = pd.concat([X, y], axis=1)
    df_resampled, _ = smote_instance.fit_resample(df, class_labels)
    return df_resampled[X_cols], df_resampled[y_cols]

def prep_data(df):
    df_ = df[fit_cols]
    
    max_delta = df_.max() - df_.min()
    delta = df_.groupby(level=in_cols) \
               .agg(lambda f: f.max() - f.min())
    slabels = (delta / max_delta >= 0.01).any(axis=1) \
                                         .to_frame('label') \
                                         .astype(int) \
                                         .reset_index(drop=True)
    hlabels = df_.groupby(level=in_cols)  \
                 .agg(lambda f: (f > 0).any() & (f <= 0).any())\
                 .any(axis=1) \
                 .to_frame('label') \
                 .astype(int) \
                 .reset_index(drop=True)
    
    df_fit = df_.copy()
    df_fit['i'] = df_fit.groupby(level=in_cols).cumcount()
    df_fit = df_fit.pivot(columns=['i'], values=fit_cols)
    df_fit.columns = [f"{f}_{i}" for f, i in df_fit.columns]

    X = df_fit.reset_index()
    one_hot = pd.get_dummies(X[enum_cols])
    X = X.drop(columns=enum_cols).join(one_hot)
    y = df_

    return X, y, slabels, hlabels

def get_last_iter(rs_df: pd.DataFrame):
    return rs_df.groupby(level=['rs_group', 'rs_iter']).last()

def evaluate(X, y, models, *, suffix=None, random_state=SEED):
    search_split = lambda sf: ( RS(sf[0], n_iter=ITER_COUNT), sf[1] )
    
    df_random_or, cnt_random_or = search_split(smartFitness(X, models=None, method='first'))
    df_smart_or, cnt_smart_or = search_split(smartFitness(X, models=models, method='first'))
    df_random_and, cnt_random_and = search_split(smartFitness(X, models=None, method='and'))
    df_smart_and, cnt_smart_and = search_split(smartFitness(X, models=models, method='and'))

    agg_func = lambda x: CSVData._aggregate(x, agg_mode=('min', 'mean')) \
                                .loc[x.index.unique()] \
                                .reset_index()

    # Random search for 10 repetitions...
    f10 = RS(agg_func(y), n_iter=ITER_COUNT)

    # Random search for 4 repetitions...
    f4 = RS(agg_func(y.groupby(level=y.index.names) \
                      .sample(4, random_state=random_state)),
            n_iter=ITER_COUNT)
    
    print('f4 - f10')
    stat_test(f4['min'], f10['min'])
    
    print('f4 - RS-Models-AND')
    stat_test(f4['min'], df_smart_and['min'])
    
    print('f4 - RS-Models-OR')
    stat_test(f4['min'], df_smart_or['min'])

    labels = ['RS-Random-AND', 'RS-Models-AND',
              'RS-Random-OR', 'RS-Models-OR',
              'RSw4REP', 'RSw10REP', 'RS']
    df_min_box = pd.concat([df_random_and['min'], df_smart_and['min'],
                            df_random_or['min'], df_smart_or['min'],
                            f4['min'], f10['min'], f10['mean']], axis=1)
    df_min_box.columns = pd.MultiIndex.from_product([labels, fit_cols])
    df_min_box = unstack_col_level(df_min_box, 'method', level=0).reset_index()
    
    plotBox(df_min_box, output_file='rs_box' + f'_{suffix}' if suffix else '' + '.pdf', show=False)

    if suffix:
        print(f"{suffix}:")

    print(f'Number of iterations for RS in random mode OR: {cnt_random_or}')
    print(f'Number of iterations for smart RS with models OR: {cnt_smart_or}')
    print(f'Number of iterations for RS in random mode AND: {cnt_random_and}')
    print(f'Number of iterations for smart RS with models AND: {cnt_smart_and}')

if __name__  == '__main__':
    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1])
    df = data.get(min_rep=EXP_REPEAT, max_rep=EXP_REPEAT, count=COUNT, random_state=SEED)
    X, y, slabels, hlabels = prep_data(df)

    smodels = train_models(X, slabels)
    evaluate(X, y, smodels, suffix='soft', random_state=SEED)

    hmodels = train_models(X, hlabels)
    evaluate(X, y, hmodels, suffix='hard', random_state=SEED)

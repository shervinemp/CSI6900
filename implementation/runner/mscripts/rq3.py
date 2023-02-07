from functools import partial
import re
import sys
import time
from typing import Any, Dict, Sequence, Tuple, Union

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
from stat_utils import stat_test
from utils import static_vars, unstack_col_level

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
    yield from (fit_range(X, i + 1) for i in rang)

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
    for X_ in fit_cum_range(X, MAX_REPEAT-1):
        scores, desc = train(X_, y, class_labels, cv=cv, **kwargs)
        f1s = scores["test_f1"]
        models.append(scores['estimator'][np.argmax(f1s)])
    return models

def train(X: pd.DataFrame, y: pd.DataFrame, class_labels: Union[pd.DataFrame, None] = None,
          method: str = 'dt', cv: int = 5, **kwargs) -> Tuple[Dict[str, Any], str]:
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

def smartFitness(X: pd.DataFrame, models: Union[Sequence[Any], None] = None, method: str = 'or',
                 max_rep: int = MAX_REPEAT, p_thresh: Union[float, None] = 0.5,
                 n_ignore: Union[int, None] = None, n_continue: Union[int, None] = None):
    if method not in ('or', 'and', 'first'):
        raise ValueError(f"Method \"{method}\" is not valid.")
    if models is None:
        if method == 'and':
            visit_proba = np.logspace(1, max_rep-1, num=max_rep-1, base=p_thresh)
        elif method == 'or':
            visit_proba = np.logspace(1, max_rep-1, num=max_rep-1, base=1-p_thresh)
        elif method == 'first':
            visit_proba = np.ones(max_rep-1) * p_thresh
        visit_proba = pd.DataFrame(visit_proba[np.newaxis, :].repeat(len(X), axis=0))
    else:
        predict = lambda m, x: m.predict(x) if hasattr(m, 'predict') else m(x)
        pred = np.array([predict(m, x) if p_thresh is None else predict(m, x) >= p_thresh \
                         for m, x in zip(models, fit_cum_range(X, max_rep-1))]).T
        pred_df = pd.DataFrame(pred)
        if method == 'and':
            visit_proba = pred_df.cumprod(axis=1)
        elif method == 'or':
            visit_proba = (~pred_df).cumprod(axis=1)
        elif method == 'first':
            visit_proba = pd.concat((pred_df[[0]],) * len(pred_df.columns), axis=1).astype(float)
    if n_ignore:
        visit_proba.loc[:, :n_ignore] = 1
    if n_continue:
        visit_proba.loc[:, n_continue:] = visit_proba[[n_continue-1]]
    visit_proba.columns = range(1, max_rep)
    w = visit_proba.copy()
    w[0] = 1
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
    cnt = (visit_proba.sum(axis=1) + 1).sum()
    
    return df, cnt

def balance_data(X, y, class_labels=None, smote_instance=SMOTE(random_state=random_)):
    if class_labels is None:
        class_labels = y
    X_cols = X.columns
    y_cols = y.columns
    df = pd.concat([X, y], axis=1)
    df_resampled, _ = smote_instance.fit_resample(df, class_labels)
    return df_resampled[X_cols], df_resampled[y_cols]

def preprocess_data(df):
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

def evaluate(X, y, models, *, suffix=None, random_state=SEED, **kwargs):
    search_split = lambda sf: ( RS(sf[0], n_iter=ITER_COUNT), sf[1] )
    
    df_random_first, cnt_random_first = search_split(smartFitness(X, models=None, method='first', **kwargs))
    df_model_first, cnt_model_first = search_split(smartFitness(X, models=models, method='first', **kwargs))
    df_random_or, cnt_random_or = search_split(smartFitness(X, models=None, method='or', **kwargs))
    df_model_or, cnt_model_or = search_split(smartFitness(X, models=models, method='or', **kwargs))
    df_random_and, cnt_random_and = search_split(smartFitness(X, models=None, method='and', **kwargs))
    df_model_and, cnt_model_and = search_split(smartFitness(X, models=models, method='and', **kwargs))

    agg_func = lambda x: CSVData._aggregate(x, agg_mode=('min', 'mean')) \
                                .loc[x.index.unique()] \
                                .reset_index()

    # Random search for 10 repetitions...
    f10 = RS(agg_func(y), n_iter=ITER_COUNT)

    # Random search for 4 repetitions...
    f4 = RS(agg_func(y.groupby(level=y.index.names) \
                      .sample(4, random_state=random_state)),
            n_iter=ITER_COUNT)

    labels = ['RS-Random-FIRST', 'RS-Model-FIRST',
              'RS-Random-OR', 'RS-Model-OR',
              'RS-Random-AND', 'RS-Model-AND',
              'RSw4REP', 'RSw10REP', 'RSw10REP-MEAN']
    
    res_arr = [df_random_first['min'], df_model_first['min'],
               df_random_or['min'], df_model_or['min'],
               df_random_and['min'], df_model_and['min'],
               f4['min'], f10['min'], f10['mean']]
    
    plotBoxMulti(res_arr, labels, suffix=suffix)

    if suffix:
        print(f"{suffix}:")
    
    rs_stats_f4 = partial(rs_stats, baseline=f4['min'], base_label='f4')
    rs_stats_f4(f10['min'], label='f10')
    for r, l in zip(res_arr[:-3], labels[:-3]):
        rs_stats_f4(r, label=l)
    
    rs_stats(df_random_first['min'], df_model_first['min'],
             labels[0], labels[1],
             cnt_random_first, cnt_model_first)
    
    rs_stats(df_random_or['min'], df_model_or['min'],
             labels[2], labels[3],
             cnt_random_or, cnt_model_or)
    
    rs_stats(df_random_and['min'], df_model_and['min'],
             labels[4], labels[5],
             cnt_random_and, cnt_model_and)

@static_vars(cl_dict=dict(zip(fit_cols, fit_labels)))
def rs_stats(results: pd.DataFrame, baseline: pd.DataFrame, label: str, base_label: str = 'baseline',
             count: Union[int, None] = None, base_count: Union[int, None] = None):
    print(f'{base_label} / {label}')
    stat_test(results, baseline, col_label_dict=rs_stats.cl_dict)
    if count and base_count:
        print(f'#iterations - {label} / {base_label}: {count} / {base_count}')
    elif count:
        print(f'#iterations - {label}: {count}')
    elif base_count:
        print(f'#iterations - {base_label}: {base_count}')

def plotBoxMulti(dfs: Sequence[pd.DataFrame], labels: Sequence[str], *, suffix=None):
    df_min_box = pd.concat(dfs, axis=1, ignore_index=True)
    df_min_box.columns = pd.MultiIndex.from_tuples([(l, c) for l, df in zip(labels, dfs) \
                                                    for c in df.columns])
    df_min_box = unstack_col_level(df_min_box, 'method', level=0).reset_index()
    
    plotBox(df_min_box, output_file='rs_box' + f'_{suffix}' if suffix else '' + '.pdf', show=False)

if __name__  == '__main__':
    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1])
    df = data.get(min_rep=EXP_REPEAT, max_rep=EXP_REPEAT, count=COUNT, random_state=SEED)
    X, y, slabels, hlabels = preprocess_data(df)

    smodels = train_models(X, slabels)
    evaluate(X, y, smodels, suffix='soft', random_state=SEED)

    hmodels = train_models(X, hlabels)
    evaluate(X, y, hmodels, suffix='hard', random_state=SEED)

    delta_model = lambda X: ((df:=X[get_fit_cols(X)]).max(axis=1) - df.min(axis=1)) >= 0.1
    dmodels = (delta_model,) * (MAX_REPEAT-1)
    evaluate(X, y, dmodels, suffix='delta', random_state=SEED, n_ignore=1)

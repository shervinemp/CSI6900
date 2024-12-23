import math
import os
import sys
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from itertools import product

from data import CSVDataLoader, balance_data, Data, make_one_hot, fit_cols

SEED = 0
MAX_REPEAT = 4

import warnings

warnings.filterwarnings("ignore")

# Create a pseudorandom number generator with the specified seed
random_ = np.random.RandomState(seed=SEED)


def fit_range(X: pd.DataFrame, rang: Union[Sequence[int], int]):
    if type(rang) is int:
        rang = range(rang)
    dcols = list(product(X.columns.levels[0], rang))
    fit_X = X[X.columns.intersection(dcols)]
    return fit_X


def prep_data(
    df: Data,
    *,
    one_hot: bool = True,
    max_repeats: Optional[int] = None,
    reset_index: bool = True,
):
    df_ = df

    if max_repeats and max_repeats > 0:
        df_ = df_.groupbyindex().head(max_repeats)

    if one_hot:
        df_ = make_one_hot(df_)

    if max_repeats != -1:
        df_ = df_.hstack_repeats()

    if reset_index:
        df_ = df_.reset_index()

    return df_


def train_cv(model, X, y, *, cv=5, random_state=None):
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model, X, y, cv=cv, scoring=["precision", "recall", "f1"], return_estimator=True
    )
    return scores


def train_model(model, X, y, *, cv=None, random_state=None):
    if cv is None:
        model.fit(X, y)
        scores = None
    else:
        scores = train_cv(model, X, y, cv=cv, random_state=random_state)
    return model, scores


def trainDT(X, y, *, cv=None, random_state=None, **kwargs):
    kwargs.setdefault("max_depth", 5)
    model = DecisionTreeClassifier(**kwargs)
    return train_model(model, X, y, cv=cv, random_state=random_state)


def trainSVM(X, y, *, cv=None, random_state=None, **kwargs):
    model = SVC(**kwargs)
    return train_model(model, X, y, cv=cv, random_state=random_state)


def trainMLP(X, y, *, cv=None, random_state=None, **kwargs):
    kwargs.setdefault("hidden_layer_sizes", (50, 100))
    kwargs.setdefault("learning_rate", "adaptive")
    model = MLPClassifier(**kwargs)
    return train_model(model, X, y, cv=cv, random_state=random_state)


def trainRF(X, y, *, cv=None, random_state=None, **kwargs):
    kwargs.setdefault("max_depth", 5)
    model = RandomForestClassifier(**kwargs)
    return train_model(model, X, y, cv=cv, random_state=random_state)


def train(
    X: pd.DataFrame,
    y: pd.DataFrame,
    class_labels: Union[pd.DataFrame, None] = None,
    method: str = "dt",
    cv: int = None,
    random_state: Union[np.random.RandomState, int, None] = None,
    **kwargs,
):
    if class_labels is None:
        class_labels = y
    X_b, y_b = balance_data(X, y, class_labels)
    y_b = y_b.iloc[:, 0]
    if method == "dt":
        model, scores = trainDT(X_b, y_b, cv=cv, random_state=random_state, **kwargs)
    elif method == "svm":
        model, scores = trainSVM(X_b, y_b, cv=cv, random_state=random_state, **kwargs)
    elif method == "mlp":
        model, scores = trainMLP(X_b, y_b, cv=cv, random_state=random_state, **kwargs)
    elif method == "rf":
        model, scores = trainRF(X_b, y_b, cv=cv, random_state=random_state, **kwargs)
    else:
        raise ValueError(f'Method "{method}" not supported.')

    return model, scores


def train_best(
    X: pd.DataFrame,
    y: pd.DataFrame,
    class_labels: Union[pd.DataFrame, None] = None,
    cv: int = 5,
    criteria: str = "test_f1",
    compare: callable = lambda a, b: a.__gt__(b),
    random_state: Union[np.random.RandomState, int, None] = None,
    **kwargs,
):
    methods = ["dt", "svm", "mlp", "rf"]
    score_best = -math.inf
    model_best = None
    for m in methods:
        model, scores = train(
            X,
            y,
            class_labels=class_labels,
            method=m,
            cv=cv,
            random_state=random_state,
            **kwargs,
        )
        score_mean = np.mean(scores[criteria])
        if compare(score_mean, score_best):
            score_best = score_mean
            model_best = model

    model_trained, _ = train_model(model_best, X, y, cv=None)

    return model_trained, score_best


def test(scores, X, y, output_file="rq3.txt"):
    models = scores["estimator"]
    preds = [model.predict(X) for model in models]
    precisions = [precision_score(y, pred) for pred in preds]
    recalls = [recall_score(y, pred) for pred in preds]
    f1s = [f1_score(y, pred) for pred in preds]
    with open(output_file, "at") as f:
        f.write(f"Approach: {str(models[0])}\n")
        f.write(f"Input: {list(X.columns)}\n")
        f.write(f"Output: {list(y.columns)}\n")
        f.write("Train:\n")
        f.write(f'  precision: {scores["test_precision"]}\n')
        f.write(f'    - mean: {scores["test_precision"].mean()}\n')
        f.write(f'  recall: {scores["test_recall"]}\n')
        f.write(f'    - mean: {scores["test_recall"].mean()}\n')
        f.write(f'  F1: {scores["test_f1"]}\n')
        f.write(f'    - mean: {scores["test_f1"].mean()}\n')
        f.write("Test:\n")
        f.write(f"  precision: {precisions}\n")
        f.write(f"    - mean: {np.mean(precisions)}\n")
        f.write(f"    - best: {np.max(precisions)}\n")
        f.write(f"  recall: {recalls}\n")
        f.write(f"    - mean: {np.mean(recalls)}\n")
        f.write(f"    - best: {np.max(recalls)}\n")
        f.write(f"  F1: {f1s}\n")
        f.write(f"    - mean: {np.mean(f1s)}\n")
        f.write(f"    - best: {np.max(f1s)}\n")
        f.write("\n")


if __name__ == "__main__":
    # Read in a list of experiments from a file specified as the first command line argument
    df_train, df_test = CSVDataLoader(sys.argv[1]).get(split=0.75)

    sl_train = df_train.get_soft_labels()
    sl_test = df_test.get_soft_labels()

    hl_train = df_train.get_hard_labels()
    hl_test = df_test.get_hard_labels()

    df_train_ = prep_data(df_train, max_repeats=MAX_REPEAT)
    df_test_ = prep_data(df_test, max_repeats=MAX_REPEAT)

    methods = ("dt", "rf", "svm", "mlp")
    mparams = ({"max_depth": 5}, {"max_depth": 5}, {}, {"max_iter": 1000})

    if os.path.exists("rq3.txt"):
        os.remove("rq3.txt")
    for method, kwparams in zip(methods, mparams):
        for i in range(MAX_REPEAT):
            X_train_ = fit_range(df_train_, i + 1)
            X_test_ = fit_range(df_test_, i + 1)
            m, s = train(X_train_, sl_train, method=method, cv=5, **kwparams)
            test(s, X_test_, sl_test, output_file=f"rq3.txt")

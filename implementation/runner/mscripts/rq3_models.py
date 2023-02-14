import os
import sys
from typing import Any, Dict, Tuple, Union, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from itertools import product

from data import CSVDataLoader, balance_data, Data, enum_cols, fit_cols

SEED = 0
MAX_REPEAT = 4

# Create a pseudorandom number generator with the specified seed
random_ = np.random.RandomState(seed=SEED)


def fit_range(X: pd.DataFrame, rang: Union[Sequence[int], int]):
    if type(rang) is int:
        rang = range(rang)
    dcols = list(product(fit_cols, rang))
    fit_X = X.drop(X.columns.intersection(dcols), axis=1)
    return fit_X


def get_X_y(df: Data):
    X = df.hstack_repeats().reset_index()
    one_hot = pd.get_dummies(X[enum_cols])
    X = X.drop(columns=enum_cols, level=0).join(one_hot)
    y = df

    return X, y


def train_model(model, X, y, *, cv=5, random_state=None):
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model, X, y, cv=cv, scoring=["precision", "recall", "f1"], return_estimator=True
    )
    desc = str(model)
    return scores, desc


def trainDecisionTree(X, y, *, cv=5, random_state=None, **kwargs):
    kwargs.setdefault("max_depth", 5)
    model = DecisionTreeClassifier(**kwargs)
    return train_model(model, X, y, cv=cv, random_state=random_state)


def trainSVM(X, y, *, cv=5, random_state=None, **kwargs):
    model = SVC(**kwargs)
    return train_model(model, X, y, cv=cv, random_state=random_state)


def trainMLP(X, y, *, cv=5, random_state=None, **kwargs):
    kwargs.setdefault("hidden_layer_sizes", (50, 100))
    kwargs.setdefault("learning_rate", "adaptive")
    model = MLPClassifier(**kwargs)
    return train_model(model, X, y, cv=cv, random_state=random_state)


def trainRF(X, y, *, cv=5, random_state=None, **kwargs):
    kwargs.setdefault("max_depth", 5)
    model = RandomForestClassifier(**kwargs)
    return train_model(model, X, y, cv=cv, random_state=random_state)


def train(
    X: pd.DataFrame,
    y: pd.DataFrame,
    class_labels: Union[pd.DataFrame, None] = None,
    method: str = "dt",
    cv: int = 5,
    random_state: Union[np.random.RandomState, int, None] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], str]:
    if class_labels is None:
        class_labels = y
    X_b, y_b = balance_data(X, y, class_labels)
    y_b = y_b.iloc[:, 0]
    if method == "dt":
        scores, desc = trainDecisionTree(
            X_b, y_b, cv=cv, random_state=random_state, **kwargs
        )
    elif method == "svm":
        scores, desc = trainSVM(X_b, y_b, cv=cv, random_state=random_state, **kwargs)
    elif method == "mlp":
        scores, desc = trainMLP(X_b, y_b, cv=cv, random_state=random_state, **kwargs)
    elif method == "rf":
        scores, desc = trainRF(X_b, y_b, cv=cv, random_state=random_state, **kwargs)
    else:
        raise ValueError(f'Method "{method}" not supported.')

    return scores, desc


def test(scores, desc, X, y, output_file="rq3.txt"):
    models = scores["estimator"]
    preds = [model.predict(X) for model in models]
    precisions = [precision_score(y, pred) for pred in preds]
    recalls = [recall_score(y, pred) for pred in preds]
    f1s = [f1_score(y, pred) for pred in preds]
    with open(output_file, "at") as f:
        f.write(f"Approach: {desc}\n")
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
    df_train, df_test = CSVDataLoader(sys.argv[1]).get(split=0.8)

    X_train, y_train = get_X_y(df_train)
    sl_train = df_train.get_soft_labels()
    hl_train = df_train.get_hard_labels()

    X_test, y_test = get_X_y(df_test)
    sl_test = df_test.get_soft_labels()
    hl_test = df_test.get_hard_labels()

    methods = ("dt", "rf", "svm", "mlp")
    mparams = ({"max_depth": 5}, {"max_depth": 5}, {}, {"max_iter": 1000})

    if os.path.exists("rq3.txt"):
        os.remove("rq3.txt")
    for method, kwparams in zip(methods, mparams):
        for i in range(MAX_REPEAT):
            X_train_ = fit_range(X_train, i + 1)
            X_test_ = fit_range(X_test, i + 1)
            m, d = train(X_train_, sl_train, method=method, **kwparams)
            test(m, d, X_test_, sl_test, output_file="rq3.txt")

import re
from typing import Sequence, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from data_utils import CSVData, enum_cols, fit_cols, in_cols
from utils import static_vars

SEED = 0
EXP_REPEAT = 10
COUNT = 1000


def get_data(csv_addr: str, *, print_len: bool = True, **kwargs):
    csv = CSVData(csv_addr)
    if print_len:
        print(f"#Entries: {len(csv)}")

    data = csv.get(
        min_rep=EXP_REPEAT, max_rep=EXP_REPEAT, count=COUNT, random_state=SEED, **kwargs
    )

    return data


def balance_data(X, y, class_labels=None, smote_instance=SMOTE(random_state=SEED)):
    if class_labels is None:
        class_labels = y
    X_cols = X.columns
    y_cols = y.columns
    df = pd.concat([X, y], axis=1)
    df_resampled, _ = smote_instance.fit_resample(df, class_labels)
    return df_resampled[X_cols], df_resampled[y_cols]


@static_vars(regs=[re.compile(f"^{f}_\d+$") for f in fit_cols])
def get_fit_cols(X):
    return [
        col for col in X.columns if np.any([r.match(col) for r in get_fit_cols.regs])
    ]


def fit_range(X, rang: Union[Sequence[int], int]):
    if type(rang) is int:
        rang = range(rang)
    dcols = list(filter(lambda f: int(f.split("_")[-1]) not in rang, get_fit_cols(X)))
    return X.drop(dcols, axis=1)


def fit_cum_range(X, rang: Union[Sequence[int], int]):
    if type(rang) is int:
        rang = range(rang)
    yield from (fit_range(X, i + 1) for i in rang)


def hstack_runs(df):
    df_fit = df.copy()
    df_fit["i"] = df_fit.groupby(level=in_cols).cumcount()
    df_fit = df_fit.pivot(columns=["i"], values=fit_cols)
    df_fit.columns = [f"{f}_{i}" for f, i in df_fit.columns]

    X = df_fit.reset_index()
    one_hot = pd.get_dummies(X[enum_cols])
    X = X.drop(columns=enum_cols).join(one_hot)
    y = df

    return X, y


def get_soft_labels(df: pd.DataFrame):
    max_delta = df.max() - df.min()
    delta = df.groupby(level=in_cols).agg(lambda f: f.max() - f.min())
    slabels = (
        (delta / max_delta >= 0.01)
        .any(axis=1)
        .to_frame("label")
        .astype(int)
        .reset_index(drop=True)
    )

    return slabels


def get_hard_labels(df: pd.DataFrame):
    hlabels = (
        df.groupby(level=in_cols)
        .agg(lambda f: (f > 0).any() & (f <= 0).any())
        .any(axis=1)
        .to_frame("label")
        .astype(int)
        .reset_index(drop=True)
    )

    return hlabels

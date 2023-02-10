import re
from typing import Sequence, Union

import numpy as np
import pandas as pd

from data_utils import CSVData, enum_cols, fit_cols, in_cols
from utils import static_vars


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


def get_X_y(df: CSVData):
    X = df.hstack_repeats().reset_index()
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

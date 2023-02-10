from itertools import product
from typing import Sequence, Union

import pandas as pd

from data_utils import Data, enum_cols, fit_cols


def fit_range(X: pd.DataFrame, rang: Union[Sequence[int], int]):
    if type(rang) is int:
        rang = range(rang)
    fcols = X[fit_cols].columns
    dcols = list(product(fit_cols, rang))
    fit_X = X.drop(fcols.intersection(dcols), axis=1)
    return fit_X


def get_X_y(df: Data):
    X = df.hstack_repeats().reset_index()
    one_hot = pd.get_dummies(X[enum_cols])
    X = X.drop(columns=enum_cols, level=0).join(one_hot)
    y = df

    return X, y

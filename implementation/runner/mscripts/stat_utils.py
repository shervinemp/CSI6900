from pprint import pprint
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import wilcoxon

from vargha_delaney import VD_A


def stat_test(a: pd.DataFrame, b: pd.DataFrame, col_label_dict=None):
    if col_label_dict is None:
        col_label_dict = {str(x): str(x) for x in a.columns.intersection(b.columns)}
    
    results = {}
    for col, label in col_label_dict.items():
        try:
            a_ = a[col].to_list()
            b_ = b[col].to_list()
            results[label] = {"wilcoxon": wilcoxon(a_, b_), "VD": VD_A(a_, b_)}
        except ValueError:
            results[label] = "Error: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements."
    pprint(results)

def delta_thresh_proba_splines(n: int, thresh_vals: Sequence[float] = np.arange(0, 1, 0.1), *, \
                               lower: float = 0., upper: float = 1., num_samples: int = 10000,
                               return_func: bool = False):
    dist = np.random.triangular(lower, (lower + upper) / 2, upper, (num_samples, n))
    dist_df = pd.DataFrame(dist)
    max_min_diff = dist_df.cummax(axis=1) - dist_df.cummin(axis=1)
    probs = pd.concat([(max_min_diff >= k).mean(axis=0) for k in thresh_vals], axis=0)
    splines = [CubicSpline(thresh_vals, probs[i]) for i in range(n)]

    if return_func:
        def func(n: int, p_thresh: float):
            return splines[n-1](p_thresh)
        return func
    
    return splines
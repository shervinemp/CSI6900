from collections import defaultdict
from pprint import pprint
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import wilcoxon
from utils import static_vars, unstack_col_level

from vargha_delaney import VD_A


@static_vars(
    tests={
        "wilcoxon": lambda a, b: dict(
            statistic=(d := wilcoxon(a, b)).statistic, pvalue=d.pvalue
        ),
        "VD": lambda a, b: dict(estimate=(d := VD_A(a, b))[0], magnitude=d[1]),
    }
)
def stat_test(
    a: pd.DataFrame,
    b: pd.DataFrame,
    col_label_dict: Optional[Dict[Union[str, int], str]] = None,
    log: bool = False,
) -> pd.DataFrame:
    if col_label_dict is None:
        col_label_dict = {str(x): str(x) for x in a.columns.intersection(b.columns)}

    results = defaultdict(dict)
    for col, label in col_label_dict.items():
        a_ = a[col].to_list()
        b_ = b[col].to_list()
        for slabel, s in stat_test.tests.items():
            try:
                d = s(a_, b_)
                for k, v in d.items():
                    results[(label, slabel, k)] = [v]
            except ValueError as e:
                pprint(
                    f'The following exception happened in "{slabel}" for "{label}":\n{str(e)}'
                )
    results = pd.DataFrame(results)
    results = unstack_col_level(results, "var", level=0)
    if log:
        with pd.option_context("display.float_format", str):
            pprint(results)

    return results


def delta_thresh_proba_splines(
    n: int,
    thresh_vals: Sequence[float] = np.arange(0, 1, 0.1),
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    num_samples: int = 10000,
    return_func: bool = False,
):
    dist = np.random.triangular(lower, (lower + upper) / 2, upper, (num_samples, n))
    dist_df = pd.DataFrame(dist)
    max_min_diff = dist_df.cummax(axis=1) - dist_df.cummin(axis=1)
    probs = pd.concat([(max_min_diff >= k).mean(axis=0) for k in thresh_vals], axis=0)
    splines = [CubicSpline(thresh_vals, probs[i]) for i in range(n)]

    if return_func:

        def func(n: int, p_thresh: float):
            return splines[n - 1](p_thresh)

        return func

    return splines

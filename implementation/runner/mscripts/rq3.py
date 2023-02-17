from pprint import pprint
import sys
from functools import partial
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from data import CSVDataLoader, fit_cols, col_label_dict
from post_rs import ITER_COUNT
from rs import RandomSearch as RS
from rq3_models import MAX_REPEAT, fit_range, get_X_y, train
from stat_utils import stat_test
from utils import hstack_with_labels, unstack_col_level, melt_multi

SEED = 0


def smart_fitness(
    X: pd.DataFrame,
    models: Union[Sequence[Any], None] = None,
    method: str = "first",
    max_rep: int = MAX_REPEAT,
    p_thresh: Union[float, None] = 0.5,
    n_ignore: Union[int, None] = None,
    n_continue: Union[int, None] = None,
):
    t_proba = get_transition_proba(
        X, models, method, max_rep, p_thresh, n_ignore, n_continue
    )
    h_proba = get_halt_proba(t_proba)

    value_vars_arr = [[(f, i) for f in fit_cols] for i in range(max_rep)]
    var_names = [f"{i}_var" for i in range(1, max_rep + 1)]
    value_names = list(range(1, max_rep + 1))
    df = melt_multi(
        X,
        value_vars_arr=value_vars_arr,
        var_names=var_names,
        value_names=value_names,
        ignore_index=False,
    )

    mean_of_rows = h_proba.mean(axis=1).to_numpy()
    normalized_h_proba = h_proba / mean_of_rows[:, np.newaxis]
    w = pd.concat([normalized_h_proba] * len(fit_cols), axis=0)
    vals = df[range(1, max_rep + 1)]

    df["min"] = (vals.cummin(axis=1) * w).mean(axis=1)
    df["mean"] = (vals.cumsum(axis=1) / range(1, vals.shape[1] + 1) * w).mean(axis=1)

    df["f"] = df["1_var"]
    df = pd.pivot(df, columns="f", values=["min", "mean"])
    cnt = (t_proba.sum(axis=1) + 1).sum()
    cnt = int(np.round(cnt))

    return df, cnt


def get_transition_proba(
    X: pd.DataFrame,
    models: Union[Sequence[Any], None] = None,
    method: str = "first",
    max_rep: int = MAX_REPEAT,
    p_thresh: Union[float, None] = 0.5,
    n_ignore: Union[int, None] = None,
    n_continue: Union[int, None] = None,
):
    if method not in ("or", "and", "first"):
        raise ValueError(f'Method "{method}" is not valid.')

    if models is None:
        if method == "and":
            t_proba = np.logspace(1, max_rep - 1, num=max_rep - 1, base=p_thresh)
        elif method == "or":
            t_proba = np.logspace(1, max_rep - 1, num=max_rep - 1, base=1 - p_thresh)
        elif method == "first":
            t_proba = np.ones(max_rep - 1) * p_thresh
        t_proba = pd.DataFrame(t_proba[np.newaxis, :].repeat(len(X), axis=0))
    else:
        predict = lambda m, x: m.predict(x) if hasattr(m, "predict") else m(x)
        pred = np.array(
            [
                predict(m, x) if p_thresh is None else predict(m, x) >= p_thresh
                for m, x in zip(models, (fit_range(X, i) for i in range(1, max_rep)))
            ]
        ).T
        pred_df = pd.DataFrame(pred)
        if method == "and":
            t_proba = pred_df.cumprod(axis=1)
        elif method == "or":
            t_proba = (~pred_df).cumprod(axis=1)
        elif method == "first":
            t_proba = pd.concat((pred_df[[0]],) * len(pred_df.columns), axis=1).astype(
                float
            )
    if n_ignore:
        t_proba.loc[:, :n_ignore] = 1
    if n_continue:
        t_proba.loc[:, n_continue:] = t_proba[[n_continue - 1]]
    return t_proba


def get_halt_proba(transition_proba: pd.DataFrame) -> pd.DataFrame:
    w = transition_proba.copy()
    pad_pos = w.shape[1] + 1
    w.columns = range(1, pad_pos)
    w[0] = 1
    w[pad_pos] = 0
    w = w.sort_index(axis=1).diff(periods=-1, axis=1).drop(columns=[pad_pos])
    return w


def train_models(X, y, class_labels=None, *, cv=5, **kwargs):
    models = []
    for X_ in (fit_range(X, i) for i in range(1, MAX_REPEAT)):
        model, scores = train(X_, y, class_labels, cv=cv, **kwargs)
        f1s = scores["test_f1"]
        models.append(scores["estimator"][np.argmax(f1s)])
    return models


def evaluate(X, y, models, *, suffix=None, random_state=SEED, **kwargs):
    search_split = lambda sf: (RS.from_dataframe(sf[0], n_iter=ITER_COUNT), sf[1])

    df_random_first, cnt_random_first = search_split(
        smart_fitness(X, models=None, method="first", **kwargs)
    )
    df_model_first, cnt_model_first = search_split(
        smart_fitness(X, models=models, method="first", **kwargs)
    )
    df_random_or, cnt_random_or = search_split(
        smart_fitness(X, models=None, method="or", **kwargs)
    )
    df_model_or, cnt_model_or = search_split(
        smart_fitness(X, models=models, method="or", **kwargs)
    )
    df_random_and, cnt_random_and = search_split(
        smart_fitness(X, models=None, method="and", **kwargs)
    )
    df_model_and, cnt_model_and = search_split(
        smart_fitness(X, models=models, method="and", **kwargs)
    )

    agg_func = (
        lambda df: df.aggregate(agg_mode=("min", "mean"))
        .loc[df.index.unique()]
        .reset_index(drop=True)
    )

    # Random search for 10 repetitions...
    f10 = RS.from_dataframe(agg_func(y), n_iter=ITER_COUNT)

    # Random search for 4 repetitions...
    f4 = RS.from_dataframe(
        agg_func(y.groupby(level=y.index.names).sample(4, random_state=random_state)),
        n_iter=ITER_COUNT,
    )

    labels = [
        "RS-Random-FIRST",
        "RS-Model-FIRST",
        "RS-Random-OR",
        "RS-Model-OR",
        "RS-Random-AND",
        "RS-Model-AND",
    ]

    base_labels = [
        "RSn4",
        "RSn10",
        "RSn10-MEAN",
    ]

    res_arr = [
        df_random_first["min"],
        df_model_first["min"],
        df_random_or["min"],
        df_model_or["min"],
        df_random_and["min"],
        df_model_and["min"],
    ]

    base_arr = [
        f4["min"],
        f10["min"],
        f10["mean"],
    ]

    res_dfs = hstack_with_labels(res_arr + base_arr, labels + base_labels)
    res_dfs = unstack_col_level(res_dfs, "method", level=0).reset_index()

    res_dfs.plot_converge_box(
        "method",
        output_file="rs_conv" + (f"_{suffix}" if suffix else "") + ".pdf",
        show=False,
    )

    res_dfs.plot(
        "method",
        output_file="rs_iters" + (f"_{suffix}" if suffix else "") + ".pdf",
        show=False,
        legend_kwargs=dict(loc="lower left", fontsize=8),
    )

    if suffix:
        print(f"{suffix}:")

    d = []
    rs_stats_f4 = partial(rs_stats, baseline=f4["min"], base_label="f4")

    s = rs_stats_f4(f10["min"], label="f10")
    d.append(s)

    for r, l in zip(res_arr, labels):
        s = rs_stats_f4(r, label=l)
        d.append(s)

    s = rs_stats(
        df_random_first["min"],
        df_model_first["min"],
        labels[0],
        labels[1],
        cnt_random_first,
        cnt_model_first,
    )
    d.append(s)

    s = rs_stats(
        df_random_or["min"],
        df_model_or["min"],
        labels[2],
        labels[3],
        cnt_random_or,
        cnt_model_or,
    )
    d.append(s)

    s = rs_stats(
        df_random_and["min"],
        df_model_and["min"],
        labels[4],
        labels[5],
        cnt_random_and,
        cnt_model_and,
    )
    d.append(s)

    stats_df = pd.concat(d, axis=0).reset_index()
    with pd.option_context("display.float_format", str):
        pprint(stats_df)


def rs_stats(
    results: pd.DataFrame,
    baseline: pd.DataFrame,
    label: str,
    base_label: str = "baseline",
    count: Union[int, None] = None,
    base_count: Union[int, None] = None,
    log: bool = False,
):
    if log:
        print(f"{base_label} / {label}")
    stats = stat_test(results, baseline, col_label_dict=col_label_dict, log=log)
    if log:
        if count and base_count:
            print(f"#iterations - {label} / {base_label}: {count} / {base_count}")
        elif count:
            print(f"#iterations - {label}: {count}")
        elif base_count:
            print(f"#iterations - {base_label}: {base_count}")
    if label:
        stats["model"] = label
    if count:
        stats["runs"] = count
    if base_label:
        stats["base_model"] = base_label
    if base_count:
        stats["base_runs"] = base_count
    return stats


if __name__ == "__main__":
    # Read in a list of experiments from a file specified as the first command line argument
    df = CSVDataLoader(sys.argv[1]).get()

    X, y = get_X_y(df)
    X_oh, _ = get_X_y(df, one_hot=True)
    slabels = df.get_soft_labels()
    hlabels = df.get_hard_labels()

    smodels = train_models(X_oh, slabels)
    evaluate(X_oh, y, smodels, suffix="soft", random_state=SEED)

    hmodels = train_models(X_oh, hlabels)
    evaluate(X_oh, y, hmodels, suffix="hard", random_state=SEED)

    delta_model = lambda X: (X.max(axis=1) - X.min(axis=1)) >= 0.1
    dmodels = (delta_model,) * (MAX_REPEAT - 1)
    for n_ignore in range(1, 10):
        evaluate(X, y, dmodels, suffix=f"delta_{n_ignore + 1}", random_state=SEED, n_ignore=n_ignore)

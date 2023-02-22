import sys
from collections import defaultdict
from functools import partial
from itertools import product
from pprint import pprint
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from data import CSVDataLoader, col_label_dict, fit_cols
from post_rs import ITER_COUNT
from rq3_models import MAX_REPEAT, fit_range, prep_data, train_best
from rs import RandomSearch as RS
from sklearn.metrics import precision_recall_fscore_support
from stat_utils import stat_test
from utils import hstack_with_labels, melt_multi, pairwise_stride2, unstack_col_level

SEED = 0

import warnings

warnings.filterwarnings("ignore")


def smart_fitness(
    X: pd.DataFrame,
    models: Optional[Sequence[Any]] = None,
    method: str = "first",
    p_thresh: float = 0.5,
    n_begin: int = 1,
    n_ignore: int = 0,
    n_end: Optional[int] = None,
    n_continue: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    i_proba = get_indiv_proba(
        X,
        models,
        p_thresh=p_thresh,
        n_begin=n_begin,
        n_ignore=n_ignore,
        n_end=n_end,
        n_continue=n_continue,
    )
    r_proba = get_reach_proba(
        i_proba,
        method=method,
    )
    h_proba = get_halt_proba(r_proba)
    n_steps = h_proba.shape[1]

    value_vars_arr = [[(f, i) for f in fit_cols] for i in range(n_steps)]
    var_names = [f"{i}_var" for i in range(n_steps)]
    value_names = range(n_steps)
    df = melt_multi(
        X,
        value_vars_arr=value_vars_arr,
        var_names=var_names,
        value_names=value_names,
        ignore_index=False,
    )

    w = pd.concat([h_proba] * len(fit_cols), axis=0)
    mean_of_rows = w.mean(axis=1).to_numpy()
    w = w / mean_of_rows[:, np.newaxis]

    vals = df[range(n_steps)]

    df["min"] = (vals.cummin(axis=1) * w).mean(axis=1)
    df["mean"] = (vals.cumsum(axis=1) / range(1, vals.shape[1] + 1) * w).mean(axis=1)

    df["f"] = df["0_var"]
    df = pd.pivot(df, columns="f", values=["min", "mean"])
    cnt = int(np.round(r_proba.sum(axis=1) + 1).sum())

    return df, cnt, h_proba


def get_indiv_proba(
    X: pd.DataFrame,
    models: Optional[Sequence[Any]] = None,
    *,
    p_thresh: bool = 0.5,
    n_begin: int = 1,
    n_ignore: int = 0,
    n_end: Optional[int] = None,
    n_continue: Optional[int] = None,
) -> pd.DataFrame:
    s = X.iloc[[0]].groupby(level=0, axis=1).size().max()

    if n_end is None:
        n_end = s
    elif n_end < 0:
        n_end = s + n_end

    predict = lambda x, m: m.predict(x) if hasattr(m, "predict") else m(x)

    if models is None:
        models = (lambda x: np.ones(x.shape[0]) * p_thresh,) * (n_end - n_begin)
    else:
        models = [
            (
                lambda p=partial(predict, m=m): (lambda x: p(x) >= p_thresh)
                if p_thresh
                else p
            )()
            for m in models
        ]

    pred = np.array(
        [
            pred_fn(x)
            for pred_fn, x in zip(
                models, (fit_range(X, i) for i in range(n_begin, n_end))
            )
        ]
    ).T
    pred_df = pd.DataFrame(pred, columns=range(n_begin, n_end))
    pred_df.loc[:, range(n_begin + n_ignore)] = 1

    if n_continue:
        pred_df.loc[:, n_begin + n_continue :] = 1

    pred_df.sort_index(axis=1, inplace=True)

    return pred_df


def get_reach_proba(
    indiv_proba: pd.DataFrame,
    method: str = "first",
) -> pd.DataFrame:
    if method not in ("or", "and", "first"):
        raise ValueError(f'Method "{method}" is not valid.')

    if method == "and":
        t_proba = indiv_proba.cumprod(axis=1)
    elif method == "or":
        t_proba = (1 - indiv_proba).cumprod(axis=1)
    elif method == "first":
        t_proba = pd.concat(
            (indiv_proba[0],) * (n_cols := indiv_proba.shape[1]),
            axis=1,
            keys=range(n_cols),
        )

    return t_proba


def get_halt_proba(reach_proba: pd.DataFrame) -> pd.DataFrame:
    h_proba = reach_proba.copy()
    pad_pos = h_proba.shape[1]
    h_proba[pad_pos] = 0
    h_proba = (
        h_proba.sort_index(axis=1).diff(periods=-1, axis=1).drop(columns=[pad_pos])
    )

    return h_proba


def train_models(X, y, class_labels=None, *, cv=5, max_repeats=MAX_REPEAT, **kwargs):
    models = []
    X_ = prep_data(X, max_repeats=max_repeats)
    t_ = partial(train_best, y=y, class_labels=class_labels, cv=cv, **kwargs)
    if max_repeats == -1:
        model, scores = t_(X_)
        return model
    for X_t in (fit_range(X_, i) for i in range(1, max_repeats + 1)):
        model, scores = t_(X_t)
        models.append(model)
    return models


def evaluate(
    X: pd.DataFrame,
    y: pd.DataFrame,
    models: Optional[Sequence[Any]] = None,
    *,
    suffix: Optional[str] = None,
    max_repeats: int = MAX_REPEAT,
    random_state=SEED,
    **kwargs,
):
    X_ = prep_data(X, max_repeats=max_repeats)

    search_split = lambda x: (RS.from_dataframe(x[0], n_iter=ITER_COUNT), *x[1:])
    sf = partial(smart_fitness, X=X_, **kwargs)

    dfs = defaultdict(dict)
    for models, method in product((None, models), ("first", "and")):
        dfs["model" if models else "random"][method] = search_split(
            sf(models=models, method=method)
        )

    agg_func = (
        lambda df: df.agg_repeats(agg_mode=("min", "mean"))
        .loc[df.index.unique()]
        .reset_index(drop=True)
    )

    # Random search for 10 repetitions...
    f10 = RS.from_dataframe(agg_func(X), n_iter=ITER_COUNT)

    # Random search for 4 repetitions...
    fm = RS.from_dataframe(
        agg_func(
            X.groupby(level=X.index.names).sample(MAX_REPEAT, random_state=random_state)
        ),
        n_iter=ITER_COUNT,
    )

    dfs_keys = list(product(("random", "model"), ("first", "and")))

    labels = [f"rs-{model}-{method}".upper() for model, method in dfs_keys]
    res_arr = [dfs[model][method][0]["min"] for model, method in dfs_keys]
    cnt_arr = [dfs[model][method][1] for model, method in dfs_keys]
    hproba_arr = [dfs[model][method][2] for model, method in dfs_keys]

    base_labels = [
        f"RSn{MAX_REPEAT}",
        "RSn10",
        "RSn10-MEAN",
    ]
    base_arr = [
        fm["min"],
        f10["min"],
        f10["mean"],
    ]
    base_cnt = [
        X_.shape[0] * X_.shape[1],
        X.shape[0] * X.shape[1],
        X.shape[0] * X.shape[1],
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

    rs_stats_fm = partial(
        rs_stats, baseline=fm["min"], base_label=base_labels[0], base_count=base_cnt[0]
    )

    d = []
    s = rs_stats_fm(f10["min"], label=base_labels[1], count=base_cnt[1])
    d.append(s)

    for r, c, l in zip(res_arr, cnt_arr, labels):
        s = rs_stats_fm(r, label=l, count=c)
        d.append(s)

    for (df1, l1, cnt1), (df2, l2, cnt2) in pairwise_stride2(
        zip(res_arr, labels, cnt_arr)
    ):
        s = rs_stats(df1, df2, l1, l2, cnt1, cnt2)
        d.append(s)

    stats_df = pd.concat(d, axis=0).reset_index(drop=True)
    with pd.option_context("display.float_format", str):
        pprint(stats_df)

    rand = (
        np.random.RandomState(random_state)
        if isinstance(random_state, int)
        else random_state
    )
    for label, proba in zip(labels, hproba_arr):
        y_pred = (rand.random(len(proba)) < proba.iloc[:, -1]).astype(int)
        metrics = precision_recall_fscore_support(y, y_pred, average="binary")
        print(f"{label}:")
        print(
            f"Precision: {metrics[0]:.3f}, Recall: {metrics[1]:.3f}, F1-score: {metrics[2]:.3f}"
        )

    print("means:")
    pprint(res_dfs.get_last_iter(groupby="method").groupby("method")[fit_cols].mean())


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
    if base_label:
        stats["base_model"] = base_label
    if count:
        stats["runs"] = count
    if base_count:
        stats["base_runs"] = base_count
    return stats


if __name__ == "__main__":
    # Read in a list of experiments from a file specified as the first command line argument
    df_train, df_test = CSVDataLoader(sys.argv[1]).get(split=0.75)

    sl_train = df_train.get_soft_labels()
    sl_test = df_test.get_soft_labels()

    hl_train = df_train.get_hard_labels()
    hl_test = df_test.get_hard_labels()

    smodels = train_models(df_train, sl_train, max_rep=MAX_REPEAT - 1)
    evaluate(df_test, sl_test, smodels, suffix="soft", random_state=SEED)

    hmodels = train_models(df_train, hl_train, max_rep=MAX_REPEAT - 1)
    evaluate(df_test, hl_test, hmodels, suffix="hard", random_state=SEED)

    delta_model = lambda X: (
        (d := X[fit_cols].groupby(level=0, axis=1)).max() - d.min()
    ).max(axis=1)
    dmodels_base = (delta_model,) * 9
    for n_ignore in range(0, 8):
        evaluate(
            df_test,
            sl_test,
            dmodels_base,
            suffix=f"delta_{n_ignore + 2}",
            max_repeats=10,
            random_state=SEED,
            p_thresh=0.1,
            n_begin=2,
            n_ignore=n_ignore,
        )

    delta_transform = (
        lambda X: (X.agg_repeats("cummax") - X.agg_repeats("cummin"))
        .max(axis=1)
        .to_frame()
    )
    dmodel = train_models(delta_transform(df_train), sl_train, max_repeats=-1)
    dmodels = (lambda X: dmodel.predict(X.groupby(level=0, axis=1).last()),) * 9
    for n_ignore in range(0, 8):
        evaluate(
            df_test,
            sl_test,
            dmodels,
            suffix=f"delta_{n_ignore + 2}",
            max_repeats=-1,
            random_state=SEED,
            p_thresh=0.1,
            n_begin=2,
            n_ignore=n_ignore,
        )

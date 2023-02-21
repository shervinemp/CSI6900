import sys
from functools import partial
from pprint import pprint
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from data import CSVDataLoader, col_label_dict, fit_cols
from post_rs import ITER_COUNT
from rq3_models import MAX_REPEAT, fit_range, prep_data, train
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
    n_beg: int = 1,
    n_ignore: Optional[int] = None,
    n_continue: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    v_proba = get_visit_proba(
        X,
        models[:-1] if models else None,
        p_thresh=p_thresh,
        n_beg=n_beg,
        n_ignore=n_ignore,
        n_continue=n_continue,
    )
    t_proba = get_transition_proba(
        v_proba,
        method=method,
    )
    h_proba = get_halt_proba(t_proba)
    n_steps = h_proba.shape[1]

    value_vars_arr = [[(f, i) for f in fit_cols] for i in range(n_steps)]
    var_names = [f"{i}_var" for i in range(1, n_steps + 1)]
    value_names = list(range(1, n_steps + 1))
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

    vals = df[range(1, n_steps + 1)]

    df["min"] = (vals.cummin(axis=1) * w).mean(axis=1)
    df["mean"] = (vals.cumsum(axis=1) / range(1, vals.shape[1] + 1) * w).mean(axis=1)

    df["f"] = df["1_var"]
    df = pd.pivot(df, columns="f", values=["min", "mean"])
    cnt_df = t_proba.sum(axis=1) + 1

    return df, cnt_df


def get_visit_proba(
    X: pd.DataFrame,
    models: Optional[Sequence[Any]] = None,
    *,
    p_thresh: bool = 0.5,
    n_beg: int = 1,
    n_end: Optional[int] = None,
    n_continue: Optional[int] = None,
) -> pd.DataFrame:
    s = X.iloc[0].groupby(level=0, axis=1).size().max()

    if n_end is None:
        n_end = s
    elif n_end < 0:
        n_end = s + n_end

    predict = lambda m, x: m.predict(x) if hasattr(m, "predict") else m(x)

    if models is None:
        models = (lambda x: np.ones(x.shape[0]) * p_thresh,) * (n_end - n_beg)
    else:
        models = [
            (
                (p := partial(predict, m=m))
                if p_thresh is None
                else (lambda x: p(x) >= p_thresh)
            )
            for m in models
        ]

    pred = np.array(
        [
            pred_fn(x)
            for pred_fn, x in zip(
                models, (fit_range(X, i) for i in range(n_beg, n_end))
            )
        ]
    ).T
    pred_df = pd.DataFrame(pred, columns=range(n_beg, n_end))
    pred_df.loc[:, range(n_beg)] = 1

    if n_continue:
        pred_df.loc[:, n_continue:] = 1

    return pred_df


def get_transition_proba(
    visit_proba: pd.DataFrame,
    method: str = "first",
) -> pd.DataFrame:
    if method not in ("or", "and", "first"):
        raise ValueError(f'Method "{method}" is not valid.')

    if method == "and":
        t_proba = visit_proba.cumprod(axis=1)
    elif method == "or":
        t_proba = (1 - visit_proba).cumprod(axis=1)
    elif method == "first":
        t_proba = pd.concat((visit_proba[[0]],) * visit_proba.shape[1], axis=1).astype(
            float
        )

    return t_proba


def get_halt_proba(transition_proba: pd.DataFrame) -> pd.DataFrame:
    h_proba = transition_proba.copy()
    pad_pos = h_proba.shape[1] + 1
    h_proba.columns = range(1, pad_pos)
    h_proba[0] = 1
    h_proba[pad_pos] = 0
    h_proba = (
        h_proba.sort_index(axis=1).diff(periods=-1, axis=1).drop(columns=[pad_pos])
    )

    return h_proba


def train_models(X, y, class_labels=None, *, cv=5, max_rep=MAX_REPEAT, **kwargs):
    models = []
    X_ = prep_data(X)
    for X_t in (fit_range(X_, i) for i in range(1, max_rep + 1)):
        model, scores = train(X_t, y, class_labels, cv=cv, **kwargs)
        f1s = scores["test_f1"]
        models.append(scores["estimator"][np.argmax(f1s)])
    return models


def evaluate(X, y, models, *, suffix=None, random_state=SEED, **kwargs):
    X_ = prep_data(X)

    search_split = lambda sf: (RS.from_dataframe(sf[0], n_iter=ITER_COUNT), sf[1])

    df_random_first, cnt_random_first = search_split(
        smart_fitness(X_, models=None, method="first", **kwargs)
    )
    df_model_first, cnt_model_first = search_split(
        smart_fitness(X_, models=models, method="first", **kwargs)
    )

    df_random_and, cnt_random_and = search_split(
        smart_fitness(X_, models=None, method="and", **kwargs)
    )
    df_model_and, cnt_model_and = search_split(
        smart_fitness(X_, models=models, method="and", **kwargs)
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

    labels = [
        "RS-Random-OR",
        "RS-Model-OR",
        "RS-Random-AND",
        "RS-Model-AND",
    ]

    base_labels = [
        f"RSn{MAX_REPEAT}",
        "RSn10",
        "RSn10-MEAN",
    ]

    res_arr = [
        df_random_first["min"],
        df_model_first["min"],
        df_random_and["min"],
        df_model_and["min"],
    ]

    count_arr = [
        cnt_random_first.sum(),
        cnt_model_first.sum(),
        cnt_random_and.sum(),
        cnt_model_and.sum(),
    ]

    base_arr = [
        fm["min"],
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
    rs_stats_f4 = partial(
        rs_stats, baseline=fm["min"], base_label=f"f{MAX_REPEAT}", base_count=len(fm)
    )

    s = rs_stats_f4(f10["min"], label="f10", count=len(f10))
    d.append(s)

    for r, c, l in zip(res_arr, count_arr, labels):
        s = rs_stats_f4(r, label=l, count=c)
        d.append(s)

    for (df1, l1, cnt1), (df2, l2, cnt2) in pairwise_stride2(
        zip(res_arr, labels, count_arr)
    ):
        s = rs_stats(df1, df2, l1, l2, cnt1, cnt2)
        d.append(s)

    stats_df = pd.concat(d, axis=0).reset_index(drop=True)
    with pd.option_context("display.float_format", str):
        pprint(stats_df)

    for l, y_pred in zip(
        labels,
        [
            cnt_random_first == 4,
            cnt_model_first == 4,
            cnt_random_and == 4,
            cnt_model_and == 4,
        ],
    ):
        print(f"{l}:")
        pprint(precision_recall_fscore_support(y, y_pred, average="binary"))

    print("means:")
    pprint(res_dfs.get_last_iter(groupby="method").groupby("method").mean())


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

    smodels = train_models(df_train, sl_train)
    evaluate(df_test, sl_test, smodels, suffix="soft", random_state=SEED)

    smodels = train_models(df_train, hl_train)
    evaluate(df_test, hl_test, smodels, suffix="hard", random_state=SEED)

    delta_model = lambda X: (
        (d := X[fit_cols].groupby(level=0, axis=1)).max() - d.min()
    ).max(axis=1)
    dmodels_base = (delta_model,) * (MAX_REPEAT - 1)
    for n_ignore in range(1, 10):
        evaluate(
            df_test,
            sl_test,
            dmodels_base,
            suffix=f"delta_{n_ignore + 1}",
            random_state=SEED,
            p_thresh=0.1,
            n_ignore=n_ignore,
        )

    delta_transform = lambda X: (X.agg_repeats("cummax") - X.agg_repeats("cummin")).max(
        axis=1
    )
    dmodels = train_models(delta_transform(df_train), sl_train)
    dmodels = [
        (lambda X: m.predict(X.groupby(level=0, axis=1).last())) for m in dmodels
    ]
    for n_ignore in range(1, 10):
        evaluate(
            df_test,
            sl_test,
            dmodels,
            suffix=f"delta_{n_ignore + 1}",
            random_state=SEED,
            p_thresh=0.1,
            n_ignore=n_ignore,
        )

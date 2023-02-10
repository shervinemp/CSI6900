import sys
from functools import partial
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from data_utils import CSVDataLoader, fit_cols, fit_labels
from post_RS import RS, get_last_iter
from rq3_data import fit_range, get_X_y
from rq3_models import MAX_REPEAT, train
from stat_utils import stat_test
from utils import hstack_with_labels, static_vars, unstack_col_level

SEED = 0
ITER_COUNT = 50


def plot_rs(df: pd.DataFrame, output_file: str = "rs_iters.pdf", *, show: bool = True):
    sns.set()
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(24, 5))
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        sns.lineplot(data=df, x="rs_iter", y=col, hue="method", ax=ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    if show:
        plt.show()


def plot_converge_vals(
    df: pd.DataFrame, output_file: str = "rs_conv.pdf", *, show: bool = True
):
    sns.set()
    data = get_last_iter(df, groupby="method")
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(24, 5))
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        sns.boxplot(data=data, x="method", y=col, showmeans=True, ax=ax)
        ax.set_ylabel(label)
        ax.legend([], [], frameon=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    if show:
        plt.show()


def smartFitness(
    X: pd.DataFrame,
    models: Union[Sequence[Any], None] = None,
    method: str = "first",
    max_rep: int = MAX_REPEAT,
    p_thresh: Union[float, None] = 0.5,
    n_ignore: Union[int, None] = None,
    n_continue: Union[int, None] = None,
):
    visit_proba = get_visit_proba(X, models, method, max_rep, p_thresh, n_ignore, n_continue)
    visit_proba.columns = range(1, max_rep)
    w = visit_proba.copy()
    w[0] = 1
    w[max_rep] = 0
    w = w.sort_index(axis=1).diff(periods=-1, axis=1).drop(columns=[max_rep])

    t = []
    for i in range(1, max_rep + 1):
        value_vars = [(f, i) for f in fit_cols]
        var_name = f"{i}_fit"
        X_melt = X.melt(
            value_vars=value_vars, var_name=var_name, value_name=i, ignore_index=False
        )
        t.append(X_melt)
    df = pd.concat(t, axis=1)

    w_df = pd.concat(
        [w / w.mean(axis=1).to_numpy()[:, np.newaxis]] * len(fit_cols), axis=0
    )
    df_vals = df[range(1, max_rep + 1)]

    df["min"] = (df_vals.cummin(axis=1) * w_df).mean(axis=1)
    df["mean"] = (df_vals.cumsum(axis=1) / range(1, df_vals.shape[1] + 1) * w_df).mean(
        axis=1
    )

    df["f"] = df["1_fit"]
    df = pd.pivot(df, columns="f", values=["min", "mean"])
    cnt = (visit_proba.sum(axis=1) + 1).sum()
    cnt = int(np.round(cnt))

    return df, cnt


def get_visit_proba(
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
            visit_proba = np.logspace(1, max_rep - 1, num=max_rep - 1, base=p_thresh)
        elif method == "or":
            visit_proba = np.logspace(
                1, max_rep - 1, num=max_rep - 1, base=1 - p_thresh
            )
        elif method == "first":
            visit_proba = np.ones(max_rep - 1) * p_thresh
        visit_proba = pd.DataFrame(visit_proba[np.newaxis, :].repeat(len(X), axis=0))
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
            visit_proba = pred_df.cumprod(axis=1)
        elif method == "or":
            visit_proba = (~pred_df).cumprod(axis=1)
        elif method == "first":
            visit_proba = pd.concat(
                (pred_df[[0]],) * len(pred_df.columns), axis=1
            ).astype(float)
    if n_ignore:
        visit_proba.loc[:, :n_ignore] = 1
    if n_continue:
        visit_proba.loc[:, n_continue:] = visit_proba[[n_continue - 1]]
    return visit_proba


def train_models(X, y, class_labels=None, *, cv=5, **kwargs):
    models = []
    for X_ in (fit_range(X, i) for i in range(1, MAX_REPEAT)):
        scores, desc = train(X_, y, class_labels, cv=cv, **kwargs)
        f1s = scores["test_f1"]
        models.append(scores["estimator"][np.argmax(f1s)])
    return models


def evaluate(X, y, models, *, suffix=None, random_state=SEED, **kwargs):
    search_split = lambda sf: (RS(sf[0], n_iter=ITER_COUNT), sf[1])

    df_random_first, cnt_random_first = search_split(
        smartFitness(X, models=None, method="first", **kwargs)
    )
    df_model_first, cnt_model_first = search_split(
        smartFitness(X, models=models, method="first", **kwargs)
    )
    df_random_or, cnt_random_or = search_split(
        smartFitness(X, models=None, method="or", **kwargs)
    )
    df_model_or, cnt_model_or = search_split(
        smartFitness(X, models=models, method="or", **kwargs)
    )
    df_random_and, cnt_random_and = search_split(
        smartFitness(X, models=None, method="and", **kwargs)
    )
    df_model_and, cnt_model_and = search_split(
        smartFitness(X, models=models, method="and", **kwargs)
    )

    agg_func = (
        lambda df: df.aggregate(agg_mode=("min", "mean"))
        .loc[df.index.unique()]
        .reset_index()
    )

    # Random search for 10 repetitions...
    f10 = RS(agg_func(y), n_iter=ITER_COUNT)

    # Random search for 4 repetitions...
    f4 = RS(
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
        "RSw4REP",
        "RSw10REP",
        "RSw10REP-MEAN",
    ]

    res_arr = [
        df_random_first["min"],
        df_model_first["min"],
        df_random_or["min"],
        df_model_or["min"],
        df_random_and["min"],
        df_model_and["min"],
        f4["min"],
        f10["min"],
        f10["mean"],
    ]

    res_dfs = hstack_with_labels(res_arr, labels)
    res_dfs = unstack_col_level(res_dfs, "method", level=0).reset_index()

    plot_converge_vals(
        res_dfs,
        output_file="rs_conv" + (f"_{suffix}" if suffix else "") + ".pdf",
        show=False,
    )

    plot_rs(
        res_dfs,
        output_file="rs_iters" + (f"_{suffix}" if suffix else "") + ".pdf",
        show=False,
    )

    if suffix:
        print(f"{suffix}:")

    rs_stats_f4 = partial(rs_stats, baseline=f4["min"], base_label="f4")
    rs_stats_f4(f10["min"], label="f10")
    for r, l in zip(res_arr[:-3], labels[:-3]):
        rs_stats_f4(r, label=l)

    rs_stats(
        df_random_first["min"],
        df_model_first["min"],
        labels[0],
        labels[1],
        cnt_random_first,
        cnt_model_first,
    )

    rs_stats(
        df_random_or["min"],
        df_model_or["min"],
        labels[2],
        labels[3],
        cnt_random_or,
        cnt_model_or,
    )

    rs_stats(
        df_random_and["min"],
        df_model_and["min"],
        labels[4],
        labels[5],
        cnt_random_and,
        cnt_model_and,
    )


@static_vars(cl_dict=dict(zip(fit_cols, fit_labels)))
def rs_stats(
    results: pd.DataFrame,
    baseline: pd.DataFrame,
    label: str,
    base_label: str = "baseline",
    count: Union[int, None] = None,
    base_count: Union[int, None] = None,
):
    print(f"{base_label} / {label}")
    stat_test(results, baseline, col_label_dict=rs_stats.cl_dict)
    if count and base_count:
        print(f"#iterations - {label} / {base_label}: {count} / {base_count}")
    elif count:
        print(f"#iterations - {label}: {count}")
    elif base_count:
        print(f"#iterations - {base_label}: {base_count}")


if __name__ == "__main__":
    # Read in a list of experiments from a file specified as the first command line argument
    df = CSVDataLoader(sys.argv[1]).get()

    X, y = get_X_y(df)
    slabels = df.get_soft_labels()
    hlabels = df.get_hard_labels()

    smodels = train_models(X, slabels)
    evaluate(X, y, smodels, suffix="soft", random_state=SEED)

    hmodels = train_models(X, hlabels)
    evaluate(X, y, hmodels, suffix="hard", random_state=SEED)

    delta_model = (
        lambda X: (X.max(axis=1) - X.min(axis=1)) >= 0.1
    )
    dmodels = (delta_model,) * (MAX_REPEAT - 1)
    evaluate(X, y, dmodels, suffix="delta", random_state=SEED, n_ignore=1)

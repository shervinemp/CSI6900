from typing import Any, Dict, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data import fit_cols, fit_labels


def random_search(df: pd.DataFrame, n_iter: int, append_index: bool = True) -> pd.DataFrame:
    g_index = pd.RangeIndex(len(df)) // n_iter
    df_ = df.groupby(g_index, as_index=False).cummin()
    df_["rs_iter"] = df.groupby(g_index).cumcount()
    df_["rs_group"] = g_index
    if append_index:
        df_.set_index(["rs_group", "rs_iter"], append=True, inplace=True)
    return df_


def get_last_iter(
    rs_df: pd.DataFrame,
    *,
    groupby: Union[int, str, Sequence[Union[int, str]], None] = None,
    as_index: bool = False,
):
    by_ = ["rs_group", "rs_iter"]
    if groupby is not None:
        by_ = [*([groupby] if isinstance(groupby, (int, str)) else groupby), *by_]
    return rs_df.groupby(by_, as_index=as_index).last()

def plot_rs(
    data: pd.DataFrame,
    class_col: Union[str, int, None] = None,
    *,
    output_file: str = "rs_iters.pdf",
    ylim_dict = None,
    legend_kwargs: Union[Dict[str, Any], None] = None,
    show: bool = False,
):
    max_iter = data["rs_iter"].max()
    sns.set(font_scale=1.0)
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 5))
    kwparams = dict(x="rs_iter", legend=False, data=data)
    if class_col:
        kwparams['hue'] = class_col
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        sns.lineplot(
            y=col, ax=ax, **kwparams
        )
        ax.set_xlim((0, max_iter))
        if type(ylim_dict) == dict:
            ax.set_ylim(*ylim_dict[col])
        ax.set_xticks(range(0, max_iter + 1, 10))
        ax.set(xlabel="iteration", ylabel=label)
        ax.margins(0)
    if type(legend_kwargs) == dict:
        fig.legend(**legend_kwargs)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(output_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_converge_box(
    data: pd.DataFrame,
    class_col: Union[str, None] = None,
    *,
    output_file: str = "rs_conv.pdf",
    ylim_dict = None,
    show: bool = False
):
    sns.set(font_scale=1.0)
    data = get_last_iter(data, groupby=(class_col or []))
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(24, 5))
    kwparams = dict(data=data, showmeans=True)
    if class_col:
        kwparams["x"] = class_col
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        sns.boxplot(y=col, ax=ax, **kwparams)
        if type(ylim_dict) == dict:
            ax.set_ylim(*ylim_dict[col])
        ax.set_ylabel(label)
        ax.legend([], [], frameon=False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.margins(0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
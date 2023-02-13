import sys
from pprint import pprint
from typing import Any, Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import CSVDataLoader, fit_cols, fit_labels
from stat_utils import stat_test
from utils import unstack_col_level

# Seed for the pseudorandom number generator
SEED = 0
# Number of experiments in each group
ITER_COUNT = 50
HIST_SIZE = 25


def RS(df: pd.DataFrame, n_iter: int, append_index: bool = True) -> pd.DataFrame:
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
    sns.set(font_scale=1.0)

    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 5))
    kwparams = dict(x="rs_iter", legend=False, data=data)
    if class_col:
        kwparams['hue'] = class_col
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        sns.lineplot(
            y=col, ax=ax, **kwparams
        )
        ax.set_xlim((0, ITER_COUNT))
        if type(ylim_dict) == dict:
            ax.set_ylim(*ylim_dict[col])
        ax.set_xticks(range(0, ITER_COUNT + 1, 10))
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

# If this script is being run as the main script
if __name__ == "__main__":
    # Create a pseudorandom number generator with the specified seed
    random_ = np.random.RandomState(seed=SEED)

    # Read in a list of experiments from a file specified as the first command line argument
    df = CSVDataLoader(sys.argv[1]).get(agg_mode=("min", "mean"))

    rs_res = RS(df, n_iter=ITER_COUNT)
    rs_res = unstack_col_level(rs_res, "agg_mode", level=0).reset_index()

    ylim_dict = dict(zip(fit_cols, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]))

    plot_rs(rs_res, "agg_mode", ylim_dict=ylim_dict, legend_kwargs=dict(labels=["RSwRep", "RS"]))

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 5))

    # Create an array of histogram bin edges
    hist_bins = np.linspace(0, ITER_COUNT, HIST_SIZE + 1)

    # Calculate the difference between non-minimum and minimum fitness values
    diff = (
        rs_res[rs_res.agg_mode == "mean"]
        .set_index(["rs_group", "rs_iter"])[fit_cols]
        .subtract(
            rs_res[rs_res.agg_mode == "min"].set_index(["rs_group", "rs_iter"])[
                fit_cols
            ]
        )
        .reset_index()
    )

    # Add a box column to the data based on the index of the data point
    diff["box"] = diff["rs_iter"].apply(
        lambda x: next(i for i, b in enumerate(hist_bins) if x < b) - 1
    )

    # Iterate over the fitness values
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        # Create a box plot of the data
        sns.boxplot(data=diff, x="box", y=col, showmeans=True, ax=ax)

        # Set the x and y labels for the plot
        ax.set(xlabel="iteration", ylabel=label)

        # Set the x-axis tick labels
        ax.set_xticks([])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig("rs_diff_plot.pdf", bbox_inches="tight")
    # Close the plot
    plt.close()

    last_iter = get_last_iter(rs_res, groupby="agg_mode")
    l_iter_min = last_iter[last_iter.agg_mode == "min"][fit_cols]
    l_iter_mean = last_iter[last_iter.agg_mode == "mean"][fit_cols]

    print("avg min:")
    pprint(l_iter_min.mean())

    print("avg mean:")
    pprint(l_iter_mean.mean())

    print("min-mean")
    stat_test(l_iter_min, l_iter_mean)

    plot_converge_box(rs_res, "agg_mode", output_file="endbox.pdf", ylim_dict=ylim_dict)

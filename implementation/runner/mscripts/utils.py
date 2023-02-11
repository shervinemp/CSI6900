from itertools import cycle
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            if callable(v):
                key = "__cached_" + k

                def getter(self):
                    value = getattr(self, key, None)
                    if value is None:
                        value = v(self)
                        setattr(self, key, value)
                    return value

                setattr(func, k, property(getter))
            else:
                setattr(func, k, v)
        return func

    return decorate


def get_level_from_index(df: pd.DataFrame):
    return list(range(df.index.nlevels))


def melt_multi(
    frame: pd.DataFrame,
    id_vars_arr=None,
    value_vars_arr=None,
    var_names=None,
    value_names=None,
    col_level=None,
    ignore_index: bool = True,
) -> pd.DataFrame:
    
    id_vars_arr = id_vars_arr or cycle([None])
    value_vars_arr = value_vars_arr or cycle([None])
    var_names = var_names or cycle([None])
    value_names = value_names or cycle([None])

    frame_multi = pd.concat([frame.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        col_level=col_level,
        ignore_index=ignore_index
    ) for id_vars, value_vars, var_name, value_name in zip(
        id_vars_arr, value_vars_arr, var_names, value_names
    )], axis=1)

    return frame_multi


def unstack_col_level(df: pd.DataFrame, var_name: str, *, level: Union[int, str]):
    df_ = df.T.unstack(level=level).T.reset_index(level=-1)
    df_ = df_.rename(columns={df_.columns[0]: var_name})
    return df_


def hstack_with_labels(dfs, labels):
    column_tuples = []
    for label, df in zip(labels, dfs):
        for column in df.columns:
            column_tuple = (
                (label, *column) if isinstance(column, tuple) else (label, column)
            )
            column_tuples.append(column_tuple)

    stacked = pd.concat(dfs, axis=1, ignore_index=True)
    stacked.columns = pd.MultiIndex.from_tuples(column_tuples)
    return stacked


def neg_histplot(
    data,
    y=None,
    hue=None,
    xlabel=None,
    ylabel=None,
    title=None,
    colors=None,
    legend_labels=None,
    bin_range=None,
    ax=None,
    return_type="axes",
):
    # Set the seaborn style
    sns.set()
    if isinstance(data, pd.DataFrame):
        # Get the data
        y_data = data[y]
        hue_data = data[hue]
        # Get the unique values in the hue column
        hue_vals = hue_data.unique()
        n_classes = len(hue_vals)
    else:
        # Get the data
        y_data = data
        hue_vals = range(len(data))
        n_classes = len(hue_vals)
    if not colors:
        # Generate a list of colors
        colors = sns.color_palette("deep", n_classes)
    if not bin_range:
        # Calculate the step for the position of the bars
        step = 1
        # Set the position of the bars on the x-axis
        pos = range(len(y_data[0]))
    else:
        # Calculate the step for the position of the bars
        step = (bin_range[1] - bin_range[0]) / len(y_data[0])
        # Set the position of the bars on the x-axis
        pos = np.linspace(bin_range[0], bin_range[1], len(y_data[0]), endpoint=False)
    # Calculate the width of the bars
    width = step / n_classes
    if ax is None:
        # Create the plot
        fig, ax = plt.subplots()
    for i, h in enumerate(hue_vals):
        if isinstance(data, pd.DataFrame):
            # Get the data for this hue value
            y_hue = y_data[hue_data == h]
        else:
            y_hue = y_data[i]
        # Plot the bars
        ax.bar(
            pos,
            y_hue,
            color=colors[i],
            width=width,
            align="edge",
            edgecolor="white",
            alpha=0.7,
            linewidth=1,
        )
        pos = [p + width for p in pos]
    # Set the x-axis range
    if bin_range:
        ax.set_xlim(bin_range)
    # Add a legend
    if legend_labels:
        # Check if the number of labels matches the number of hue values
        if len(legend_labels) != len(hue_vals):
            raise ValueError(
                "Number of legend labels does not match number of hue values"
            )
    elif isinstance(data, pd.DataFrame):
        # Use the hue values as the legend labels
        legend_labels = hue_vals
    if legend_labels:
        ax.legend(legend_labels)
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # Use seaborn's despine function to remove the top and right spines
    sns.despine()

    if return_type == "axes":
        return ax
    elif return_type == "fig":
        return fig

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from vargha_delaney import VD_A


def static_vars(**kwargs):
    def decorate(func):
        for k, v in kwargs.items():
            setattr(func, k, v)
        return func
    return decorate

def unstack_col_level(df, var_name, *, level):
    df_ = df.T \
            .unstack(level=level) \
            .T \
            .reset_index(level=-1)
    df_ = df_.rename(columns={df_.columns[0]: var_name})
    return df_

def stat_test(a: pd.DataFrame, b: pd.DataFrame):
    print("wilcoxon:")
    pprint({label: wilcoxon(a[col].to_list(), b[col].to_list()) \
           for col, label in zip(fit_cols, fit_labels)})
    
    print("VD:")
    pprint({label: VD_A(a[col].to_list(), b[col].to_list()) \
           for col, label in zip(fit_cols, fit_labels)})

def neg_histplot(data, y=None, hue=None, xlabel=None, ylabel=None, title=None, colors=None, legend_labels=None, bin_range=None, ax=None, return_type='axes'):
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
        ax.bar(pos, y_hue, color=colors[i], width=width, align='edge', edgecolor='white', alpha=0.7, linewidth=1)
        pos = [p + width for p in pos]
    # Set the x-axis range
    if bin_range:
        ax.set_xlim(bin_range)
    # Add a legend
    if legend_labels:
        # Check if the number of labels matches the number of hue values
        if len(legend_labels) != len(hue_vals):
            raise ValueError("Number of legend labels does not match number of hue values")
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
    
    if return_type == 'axes':
        return ax
    elif return_type == 'fig':
        return fig

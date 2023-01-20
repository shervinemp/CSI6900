import itertools as it
from bisect import bisect_left
from functools import cached_property
from glob import glob
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from pandas.core.groupby.generic import DataFrameGroupBy

in_cols = [
        "Road type",
        "Road ID",
        "Scenario Length", 
        "Vehicle_in_front", 
        "vehicle_in_adjcent_lane", 
        "vehicle_in_opposite_lane", 
        "vehicle_in_front_two_wheeled", 
        "vehicle_in_adjacent_two_wheeled", 
        "vehicle_in_opposite_two_wheeled",
        "time of day", 
        "weather",
        "Number of People", 
        "Target Speed", 
        "Trees in scenario", 
        "Buildings in Scenario", 
        "task"
]

enum_cols = [in_cols[i] for i in [0, 1, 10, 15]]

fit_cols = [f'f{i}' for i in [1, 2, 4, 5]]
fit_labels = ['Distance from center lane (dfc)',
              'Distance from non-ego vehicles (dfv)',
              'Distance from static objects (dfo)',
              'Distance travelled (dt)']

def get_fv_files(fv):
    fv_ = [x for x in fv]
    fv_[-4] = fv_[-4] * 10
    return list(filter(lambda x: (x[-4] != '.' and x[-5] != '.'),
                       glob(f'[[]{", ".join(map(str, fv_))}[]]*')))


class CSVData:
    def __init__(self, filepath, min_run=0, max_run=None):
        self._df = pd.read_csv(filepath, index_col=0)
        # Set the "Road type" to "task" columns as the index of the DataFrame
        self._df.set_index(in_cols, inplace=True)
        # Group the _data by the index and apply a custom function to filter the groups
        if min_run > 0:
                self._df = self.group_by_index().filter(lambda group: group.shape[0] >= min_run)
        if max_run is not None:
                self._df = self._df[self._df.run <= max_run]
        # Sort the index
        self._df.sort_index(inplace=True)
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()
    
    def group_by_index(self) -> DataFrameGroupBy:
        return self.df.groupby(level=self.df.index.names)
    
    @cached_property
    def indices(self, to_list=False) -> list:
        # Convert the index to a list of tuples
        return self.df.index.unique().to_flat_index()
    
    def __len__(self) -> int:
        # Return the number of indices of the DataFrame
        return len(self.indices)


def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000

    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/

    :param treatment: a numeric list
    :param control: another numeric list

    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    """

    :param data: pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.
    :param val_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains values.
    :param group_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains group names.
    :param sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.

    :return: stats : pandas DataFrame of effect sizes

    Stats summary ::
    'A' : Name of first measurement
    'B' : Name of second measurement
    'estimate' : effect sizes
    'magnitude' : magnitude

    """

    x = data.copy()
    if sort:
        x[group_col] = pd.Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)

    groups = x[group_col].unique()

    # Pairwise combinations
    g1, g2 = np.array(list(it.combinations(np.arange(groups.size), 2))).T

    # Compute effect size for each combination
    ef = np.array([VD_A(list(x[val_col][x[group_col] == groups[i]].values),
                        list(x[val_col][x[group_col] == groups[j]].values)) for i, j in zip(g1, g2)])

    return pd.DataFrame({
        'A': np.unique(data[group_col])[g1],
        'B': np.unique(data[group_col])[g2],
        'estimate': ef[:, 0],
        'magnitude': ef[:, 1]
    })

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

from functools import cached_property
from glob import glob
import pandas as pd
import re

val_cols = [
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

# fit_cols = ['DE', 'DfC', 'DfV', 'DfP', 'DfM', 'DT']
fit_cols = [f'f{i}' for i in [1, 2, 4, 5]]
fit_labels = ['Dist Center', 'Dist Vehicle', 'Dist Static', 'Dist Travel']

def get_fv_files(fv):
        fv_ = [x for x in fv]
        fv_[-4] = fv_[-4] * 10
        return list(filter(lambda x: (x[-4] != '.' and x[-5] != '.'),
                           glob(f'[[]{", ".join(map(str, fv_))}[]]*')))

def separate_suffix(string):
        # Use a regular expression to match the last group of digits in the string
        match = re.search(r'\d+$', string)
        if match:
                # Return the matched digits as a string
                return int(match.group())
        else:
                # Return None if no digits were found
                return None


class CSVData:
        def __init__(self, filepath, min_run=0, max_run=None):
                self._data = pd.read_csv(filepath, index_col=0)
                # Set the "Road type" to "task" columns as the index of the DataFrame
                self._data.set_index(list(self._data.columns[:-7]), inplace=True)
                # Group the _data by the index and apply a custom function to filter the groups
                if min_run > 0:
                        self._data = self.group_by_index().filter(lambda group: group.shape[0] >= min_run)
                if max_run is not None:
                        self._data = self.filter_max_run(max_run)
                # Sort the index
                self._data.sort_index(inplace=True)
        
        def filter_max_run(self, max_run):
                return self._data[self._data.run <= max_run]

        def get_f_values(self, values):
                # Select rows using the specified values as the index
                _data = self._data.loc[tuple(values)]
                # Extract the _data for the "f1" to "f6" columns and return as a list of tuples
                return [tuple(row) for row in _data[_data.columns[-7:-1]].values]
        
        def group_by_index(self):
                return self._data.groupby(level=self._data.index.names)
        
        @cached_property
        def indices(self):
                # Convert the index to a list of tuples
                return self._data.index.unique().to_flat_index()
        
        @property
        def size(self):
                # Return the number of indices of the DataFrame
                return len(self.indices)

import itertools as it

from bisect import bisect_left
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as ss

from pandas import Categorical


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
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
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

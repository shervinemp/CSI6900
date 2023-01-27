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
    def __init__(self, filepath):
        self._df = pd.read_csv(filepath, index_col=0)
        # Set the "Road type" to "task" columns as the index of the DataFrame
        self._df.set_index(in_cols, inplace=True)
        # Sort the index
        self._df.sort_index(inplace=True)
    
    @property
    def df(self) -> pd.DataFrame:
        return self.get()
    
    def get(self, count: int = None, min_rep: int = None,
            max_rep: int = None, random_state=None) -> pd.DataFrame:
        df = self._df
        if min_rep:
            df = df.groupby(level=in_cols) \
                   .filter(lambda group: group.shape[0] >= min_rep)
        if max_rep:
            df = df.groupby(level=in_cols) \
                   .sample(max_rep, random_state=random_state)
        if count:
            df = df.loc[df.index.unique().to_series().sample(count, random_state=random_state)]
        return df.copy()
    
    @cached_property
    def indices(self, to_list=False) -> list:
        # Convert the index to a list of tuples
        return self.df.index.unique().to_flat_index()
    
    def __len__(self) -> int:
        # Return the number of indices of the DataFrame
        return len(self.indices)

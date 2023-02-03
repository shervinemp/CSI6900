from functools import cached_property, reduce
from glob import glob
from typing import Iterable, Union

import numpy as np
import pandas as pd

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
        # Set the index of the DataFrame
        self._df.set_index(in_cols, inplace=True)
        # Sort the index
        self._df.sort_index(inplace=True)
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()
    
    def get(self, count: Union[int, None] = None,
            min_rep: Union[int, None] = None, max_rep: Union[int, None] = None,
            randomize: bool = True, random_state: Union[int, np.random.RandomState, None] = None,
            agg_mode: Union[str, Iterable[str], None] = None,
            split: Union[float, None] = None, agg_test_split: bool = False) -> pd.DataFrame:
        
        df = self._df
        if min_rep:
            df = df.groupby(level=in_cols) \
                   .filter(lambda group: group.shape[0] >= min_rep)
        if max_rep:
            df = df.groupby(level=in_cols) \
                   .sample(max_rep, random_state=random_state)
        if count or randomize:
            if count is None:
                count = self.__len__()
            df = CSVData._sample_index(df, count, return_sorted=(randomize is False), random_state=random_state)
        
        if split:
            df = ( (train:=CSVData._sample_index(df, int(count * split), random_state=random_state)),
                   df.drop(train.index.to_list()) )
        if agg_mode:
            agg_func = lambda x: CSVData._aggregate(df=x, agg_mode=agg_mode, randomize=randomize, random_state=random_state)
            if split:
                if agg_test_split:
                    df = (agg_func(df[0]), agg_func(df[1]))
                else:
                    df = (agg_func(df[0]), df[1])
            else:
                df = agg_func(df)
        
        return df
    
    @cached_property
    def indices(self) -> list:
        return self.df.index.unique()
    
    def __len__(self) -> int:
        # Return the number of indices of the DataFrame
        return len(self.indices)
    
    @staticmethod
    def _sample_index(df: pd.DataFrame, count: int, *, return_sorted: bool = False,
                     random_state: Union[int, np.random.RandomState, None] = None):
        sample =  df.loc[df.index.unique().to_series().sample(count, random_state=random_state)]
        if return_sorted:
            sample = sample.sort_index()
        return sample
    
    @staticmethod
    def _aggregate(df: pd.DataFrame, agg_mode: Union[str, Iterable[str]], *,
                   randomize: bool = False, random_state: Union[int, np.random.RandomState, None] = None):
        
        if isinstance(agg_mode, str):
            multi_agg = False
            agg_mode = (agg_mode,)
        else:
            multi_agg = True
        
        repeats = df.groupby(level=list(range(df.index.nlevels)))
        if randomize:
            repeats = repeats.sample(frac=1, random_state=random_state) \
                             .groupby(level=list(range(df.index.nlevels)))
        
        df_ = reduce(lambda l, r: pd.merge(l, r, left_index=True, right_index=True),
                     map(lambda func: repeats.agg(func), agg_mode))
        if multi_agg:
            if df.columns.nlevels == 1:
                ncols = pd.MultiIndex.from_product([agg_mode, df.columns])
            else:
                ncols = pd.MultiIndex.from_tuples([(a, *c) for a in agg_mode for c in df.columns])
            df_.columns = ncols
        
        return df_

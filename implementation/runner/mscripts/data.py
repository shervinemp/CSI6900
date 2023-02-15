from __future__ import annotations

from functools import cached_property, reduce
from glob import glob
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from utils import get_level_from_index

SEED = 0
EXP_REPEAT = 10
COUNT = 1000

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
    "task",
]

enum_cols = [in_cols[i] for i in [0, 1, 10, 15]]

fit_cols = [f"f{i}" for i in [1, 2, 4, 5]]
fit_labels = [
    "Distance from center lane (dfc)",
    "Distance from non-ego vehicles (dfv)",
    "Distance from static objects (dfo)",
    "Distance travelled (dt)",
]


def get_fv_files(fv):
    fv_ = [x for x in fv]
    fv_[-4] = fv_[-4] * 10
    return list(
        filter(
            lambda x: (x[-4] != "." and x[-5] != "."),
            glob(f'[[]{", ".join(map(str, fv_))}[]]*'),
        )
    )


def balance_data(X: pd.DataFrame, y: pd.DataFrame, class_labels: Optional[str] = None, smote_instance=None):
    if smote_instance is None:
        smote_instance = SMOTE(random_state=SEED)
    if class_labels is None:
        class_labels = y
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    X_cols = X.columns
    y_cols = y.columns
    X.columns = range(X.shape[1])
    y.columns = range(y.shape[1])
    df = pd.concat([X, y], axis=1)
    df_resampled, _ = smote_instance.fit_resample(df, class_labels)
    X_resampled = df_resampled.iloc[:, :X.shape[1]]
    y_resampled = df_resampled.iloc[:, X.shape[1]:]
    X_resampled.columns = X_cols
    y_resampled.columns = y_cols
    return X_resampled, y_resampled


class Data(pd.DataFrame):
    def __init__(self, data: Union[pd.DataFrame, Data], *args, **kwargs):
            if (
                kwargs.get("copy") is None
                and isinstance(data, pd.DataFrame)
                and not isinstance(data, Data)
            ):
                kwargs.update(copy=True)
            super().__init__(data, *args, **kwargs)
    
    @property
    def _constructor(self):
        return Data
    
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self)
    
    def get_soft_labels(self, thresh=0.01) -> pd.DataFrame:
        max_delta = self.max() - self.min()
        delta = self.groupby(level=get_level_from_index(self)).agg(lambda f: f.max() - f.min())
        slabels = (
            (delta / max_delta >= thresh)
            .any(axis=1)
            .to_frame("label")
            .astype(int)
            .reset_index(drop=True)
        )

        return slabels


    def get_hard_labels(self) -> pd.DataFrame:
        hlabels = (
            self.groupby(level=get_level_from_index(self))
            .agg(lambda f: (f > 0).any() & (f <= 0).any())
            .any(axis=1)
            .to_frame("label")
            .astype(int)
            .reset_index(drop=True)
        )

        return hlabels
    
    def hstack_repeats(self, inplace: bool = False) -> Data:
        df_ = self if inplace else self.copy()
        cols = df_.columns
        df_["i"] = df_.groupby(level=get_level_from_index(self)).cumcount()
        df_ = df_.pivot(columns=["i"], values=cols)

        return df_.to_df()

    def sample_by_index(
        self,
        count: int,
        *,
        return_sorted: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> Data:
        sample = self.loc[
            self.index.unique().to_series().sample(count, random_state=random_state)
        ]
        if return_sorted:
            sample = sample.sort_index()
        return sample

    def aggregate(
        self,
        agg_mode: Union[str, Iterable[str]],
        *,
        randomize: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> Data:

        if isinstance(agg_mode, str):
            multi_agg = False
            agg_mode = (agg_mode,)
        else:
            multi_agg = True

        repeats = self.groupby(level=get_level_from_index(self))
        if randomize:
            repeats = repeats.sample(frac=1, random_state=random_state).groupby(
                level=get_level_from_index(self)
            )

        df_ = reduce(
            lambda l, r: l.merge(r, left_index=True, right_index=True),
            map(lambda func: repeats.agg(func), agg_mode),
        )
        if multi_agg:
            if self.columns.nlevels == 1:
                ncols = pd.MultiIndex.from_product([agg_mode, self.columns])
            else:
                ncols = pd.MultiIndex.from_tuples(
                    [(a, *c) for a in agg_mode for c in self.columns]
                )
            df_.columns = ncols

        return df_


class CSVDataLoader:
    def __init__(self, filepath, index=in_cols):
        self._df = pd.read_csv(filepath, index_col=0)
        # Set the index of the DataFrame
        self._df.set_index(index, inplace=True)
        # Sort the index
        self._df.sort_index(inplace=True)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    def get(
        self,
        count: int = COUNT,
        min_rep: int = EXP_REPEAT,
        max_rep: int = EXP_REPEAT,
        columns: Sequence[Union[str, int]] = fit_cols,
        randomize: bool = True,
        random_state: Union[int, np.random.RandomState] = SEED,
        agg_mode: Optional[Union[str, Iterable[str]]] = None,
        split: Optional[float] = None,
        agg_test_split: bool = False,
    ) -> pd.DataFrame:

        df = Data(self._df)[columns]
        if min_rep:
            df = df.groupby(level=get_level_from_index(df)).filter(
                lambda group: group.shape[0] >= min_rep
            )
        if max_rep:
            df = df.groupby(level=get_level_from_index(df)).sample(max_rep, random_state=random_state)
        if count or randomize:
            if count is None:
                count = self.__len__()
            df = df.sample_by_index(
                count, return_sorted=(randomize is False), random_state=random_state
            )

        if split:
            df = (
                (
                    train := df.sample_by_index(
                        int(count * split), random_state=random_state
                    )
                ),
                df.drop(train.index.to_list()),
            )
        if agg_mode:
            agg_func = lambda df: df.aggregate(
                agg_mode=agg_mode, randomize=randomize, random_state=random_state
            )
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
    def load_data(csv_addr: str, *, print_len: bool = True, **kwargs):
        csv = CSVDataLoader(csv_addr)
        if print_len:
            print(f"#Entries: {len(csv)}")

        data = csv.get(**kwargs)

        return data

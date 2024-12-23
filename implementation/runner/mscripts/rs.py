from __future__ import annotations

from itertools import cycle
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FONT_SCALE = 1.5


class RandomSearch(pd.DataFrame):
    iter_col = "_rs_iter"
    group_col = "_rs_group"

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame, n_iter: int, append_index: bool = True
    ) -> RandomSearch:
        g_index = pd.RangeIndex(len(df)) // n_iter
        df_ = df.groupby(g_index, as_index=False).cummin()
        df_[RandomSearch.iter_col] = df.groupby(g_index).cumcount()
        df_[RandomSearch.group_col] = g_index
        if append_index:
            df_.set_index(
                [RandomSearch.group_col, RandomSearch.iter_col],
                append=True,
                inplace=True,
            )
        return RandomSearch(df_)

    @property
    def _constructor(self):
        return RandomSearch

    def get_last_iter(
        self,
        *,
        groupby: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        as_index: bool = False,
    ) -> RandomSearch:
        by_ = [RandomSearch.group_col, RandomSearch.iter_col]
        if groupby is not None:
            by_ = [*([groupby] if isinstance(groupby, (int, str)) else groupby), *by_]
        return self.groupby(by_, as_index=as_index).last()

    def plot(
        self,
        class_col: Optional[Union[str, int]] = None,
        *,
        labels: Optional[Sequence[str]] = None,
        output_file: str = "rs_iters.pdf",
        ylim_dict: Optional[Dict[Union[str, int], Tuple[float, float]]] = None,
        legend_kwargs: Optional[Dict[str, Any]] = None,
        vertical: bool = True,
        show: bool = False,
    ):
        labels = labels or cycle([None])
        legend_kwargs = legend_kwargs or {}
        if class_col is not None:
            class_col_labels = self[class_col].unique()
            legend_kwargs.setdefault("title", class_col)
            legend_kwargs.setdefault("labels", class_col_labels)
        cols = self.columns.difference(
            [RandomSearch.group_col, RandomSearch.iter_col]
            + ([] if class_col is None else [class_col])
        )
        mi_iter, ma_iter = (
            it := self.reset_index()[RandomSearch.iter_col]
        ).min(), it.max()
        data = self.reset_index()

        sns.set(font_scale=FONT_SCALE)
        
        figsize = (5, 5 * len(cols)) if vertical else (5 * len(cols), 5)
        fig, axes = (
            plt.subplots(len(cols), 1, figsize=figsize)
            if vertical
            else plt.subplots(1, len(cols), figsize=figsize)
        )

        kwparams = dict(x=RandomSearch.iter_col, legend=False, data=data)
        if class_col:
            kwparams["hue"] = class_col
        
        ax_iter = axes if vertical else axes.flat
        for ax, col, label in zip(ax_iter, cols, labels):
            sns.lineplot(y=col, ax=ax, **kwparams)
            ax.set_xlim((mi_iter, ma_iter))
            if type(ylim_dict) == dict:
                ax.set_ylim(*ylim_dict[col])
            ax.set_xticks(range(mi_iter, ma_iter + 1, 10))
            ax.set(xlabel="iteration", ylabel=label)
            ax.margins(0)
        
        if legend_kwargs:
            fig.legend(**legend_kwargs)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(output_file, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def plot_box(
        self,
        count_hist: int,
        class_col: Optional[Union[str, int]] = None,
        *,
        labels: Optional[Sequence[str]] = None,
        output_file: str = "rs_iters_box.pdf",
        ylim_dict: Optional[Dict[Union[str, int], Tuple[float, float]]] = None,
        legend_kwargs: Optional[Dict[str, Any]] = None,
        vertical: bool = True,
        show: bool = False,
    ):
        labels = labels or cycle([None])
        legend_kwargs = legend_kwargs or {}
        if class_col is not None:
            class_col_labels = self[class_col].unique()
            legend_kwargs.setdefault("title", class_col)
            legend_kwargs.setdefault("labels", class_col_labels)
        cols = self.columns.difference(
            [RandomSearch.group_col, RandomSearch.iter_col]
            + ([] if class_col is None else [class_col])
        )
        mi_iter, ma_iter = (
            it := self.reset_index()[RandomSearch.iter_col]
        ).min(), it.max()
        hist_bins = np.linspace(mi_iter, ma_iter + 1, count_hist + 1)[1:]
        data = self.reset_index()
        data["box"] = data.reset_index()[RandomSearch.iter_col].apply(
            lambda x: next(i for i, b in enumerate(hist_bins) if x < b)
        )

        sns.set(font_scale=FONT_SCALE)

        figsize = (5, 5 * len(cols)) if vertical else (5 * len(cols), 5)
        fig, axes = (
            plt.subplots(len(cols), 1, figsize=figsize)
            if vertical
            else plt.subplots(1, len(cols), figsize=figsize)
        )

        kwparams = dict(x=RandomSearch.iter_col, legend=False, data=data)
        if class_col:
            kwparams["hue"] = class_col

        ax_iter = axes if vertical else axes.flat
        for ax, col, label in zip(ax_iter, cols, labels):
            sns.boxplot(data=data, x="box", y=col, showmeans=True, ax=ax)
            ax.set_xlim((mi_iter, ma_iter))
            if type(ylim_dict) == dict:
                ax.set_ylim(*ylim_dict[col])
            ax.set_xticks(range(mi_iter, ma_iter + 1, 10))
            ax.set(xlabel="iteration", ylabel=label)
            ax.margins(0)

        if legend_kwargs:
            fig.legend(**legend_kwargs)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(output_file, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def plot_converge_box(
        self,
        class_col: Optional[str] = None,
        *,
        labels: Optional[Sequence[str]] = None,
        output_file: str = "rs_conv.pdf",
        ylim_dict: Optional[Dict[Union[str, int], Tuple[float, float]]] = None,
        vertical: bool = True,
        show: bool = False,
    ):
        labels = labels or cycle([None])
        cols = self.columns.difference(
            [RandomSearch.group_col, RandomSearch.iter_col]
            + ([] if class_col is None else [class_col])
        )

        sns.set(font_scale=FONT_SCALE)

        data = self.get_last_iter(groupby=(class_col or [])).reset_index()

        figsize = (5, 5 * len(cols)) if vertical else (5 * len(cols), 5)
        fig, axes = (
            plt.subplots(len(cols), 1, figsize=figsize)
            if vertical
            else plt.subplots(1, len(cols), figsize=figsize)
        )

        kwparams = dict(data=data, showmeans=True)
        if class_col:
            kwparams["x"] = class_col
        
        ax_iter = axes if vertical else axes.flat
        for ax, col, label in zip(ax_iter, cols, labels):
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

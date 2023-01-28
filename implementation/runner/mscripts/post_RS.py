import sys
from functools import partial, reduce
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from data_utils import CSVData, fit_cols, fit_labels
from vargha_delaney import VD_A

# Seed for the pseudorandom number generator
SEED = 0

# Number of experiments in each group
ITER_COUNT = 50

# Maximum number of groups to create (if not None)
RS_REPEAT = 20

# Maximum number of experiment repeats
EXP_REPEAT = 10

HIST_SIZE = 25

# If this script is being run as the main script
def RS(df, n_iter, agg_mode='mean', randomize=True, random_state=None):
    def agg_op(agg_mode, *, group):
        if agg_mode == 'first':
            return group.first()
        elif agg_mode == 'min':
            return group.min()
        elif agg_mode == 'mean':
            return group.mean()
        else:
            raise ValueError(f"agg_mode \"{agg_mode}\" not supported.")

    if type(agg_mode) in (list, tuple):
        multi_agg = True
    else:
        multi_agg = False
        agg_mode = (agg_mode,)

    if random_state is None or type(random_state) is int:
        random_state = np.random.RandomState(random_state)
    
    df_ = df.copy()
    indices = df.index.unique()
    
    if randomize is True:
        df_ = df_.loc[indices.to_series().sample(frac=1, random_state=random_state)]

    repeats = df_.groupby(level=list(range(df_.index.nlevels))).size()
    index_group = np.array([x for x, r in zip(random_state.permutation(len(indices)), repeats) \
                              for _ in range(r)])
    df_['group_id'], df_['x'] = divmod(index_group, n_iter)
    df_.set_index(['group_id', 'x'], append=True, inplace=True)
    
    val_grp = df_.groupby(level=['group_id', 'x'])
    values_df = reduce(lambda l, r: pd.merge(l, r, left_index=True, right_index=True),
                       map(partial(agg_op, group=val_grp), agg_mode))
    if multi_agg:
        if df.columns.nlevels == 1:
            ncols = pd.MultiIndex.from_product([agg_mode, df.columns])
        else:
            ncols = pd.MultiIndex.from_tuples([(a, *c) for a in agg_mode for c in df.columns])
        values_df.columns = ncols
    
    res = values_df.groupby(level=['group_id']) \
                   .cummin()
    
    return res

if __name__ == '__main__':
    # Create a pseudorandom number generator with the specified seed
    random_ = np.random.RandomState(seed=SEED)

    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1])
    print(f"#Entries: {len(data)}")

    n_scene = ITER_COUNT * RS_REPEAT
    df = data.get(min_rep=EXP_REPEAT, max_rep=EXP_REPEAT, count=n_scene, random_state=SEED)
    
    rs_res = RS(df[fit_cols], n_iter=ITER_COUNT, agg_mode=('min', 'mean'), random_state=random_)
    rs_res = rs_res.T \
                   .unstack(level=0) \
                   .T \
                   .reset_index(level=2) \
                   .rename(columns={'level_2': 'agg_mode'}) \
                   .reset_index()

    ylim_dict = dict(zip(fit_cols, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]))

    # Set the font scale for seaborn plots
    sns.set(font_scale=1.0)

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 5))
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        # Create a line plot of the data for the fitness value
        g = sns.lineplot(x='x', y=col, hue='agg_mode', legend=False,
                         data=rs_res, ax=ax)
        
        ax.set_xlim((0, 50))
        ax.set_ylim(*ylim_dict[col])

        ax.set_xticks(range(0, 51, 10))

        # Set the x and y labels for the plot
        ax.set(xlabel='iteration', ylabel=label)

        ax.margins(0)
    # Set the legend labels
    fig.legend(labels=['RSwRep', 'RS'])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('rs_plot.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 5))

    # Create an array of histogram bin edges
    hist_bins = np.linspace(0, ITER_COUNT, HIST_SIZE + 1)
    
    # Calculate the difference between non-minimum and minimum fitness values
    diff = rs_res[rs_res.agg_mode == 'mean'].set_index(['group_id', 'x'])[fit_cols] \
           .subtract(rs_res[rs_res.agg_mode == 'min'].set_index(['group_id', 'x'])[fit_cols]) \
           .reset_index()
    
    # Add a box column to the data based on the index of the data point
    diff['box'] = diff['x'].apply(
        lambda x: next(i for i, b in enumerate(hist_bins) if x < b) - 1
    )

    # Iterate over the fitness values
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        # Create a box plot of the data
        sns.boxplot(x='box', y=col, data=diff, showmeans=True, ax=ax)
        
        # Set the x and y labels for the plot
        ax.set(xlabel='iteration', ylabel=label)
        
        # Set the x-axis tick labels
        ax.set_xticks([])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('rs_diff_plot.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

    last_iter = rs_res.groupby(['group_id', 'agg_mode', 'x']).last().reset_index()
    b_ = last_iter[last_iter.agg_mode == 'min']
    c_ = last_iter[last_iter.agg_mode == 'mean']
    
    df_end = pd.concat([b_, c_])

    print('avg min:')
    pprint(b_[fit_cols].mean())

    print('avg mean:')
    pprint(c_[fit_cols].mean())

    print("min-mean")
    pprint({label: wilcoxon(c_[col].to_list(), b_[col].to_list()) \
           for col, label in zip(fit_cols, fit_labels)})
    
    pprint({label: VD_A(c_[col].to_list(), b_[col].to_list()) \
           for col, label in zip(fit_cols, fit_labels)})
    
    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(4 * len(fit_cols), 4))

    # Iterate over the fitness values
    for ax, col, label in zip(axes, fit_cols, fit_labels):        
        # Create a histogram of the data
        sns.boxplot(data=df_end, x='agg_mode', y=col, orient='v', showmeans=True, ax=ax)
        
        # Set the x and y labels for the plot
        ax.set(xlabel="aggregation", ylabel=label)

        ax.set_xticklabels(['RSwRep', 'RS'])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('end_box.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

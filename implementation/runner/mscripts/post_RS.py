import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from data_utils import CSVData, fit_cols, fit_labels
from utils import unstack_col_level
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

def RS(df: pd.DataFrame, n_iter: int, append_index: bool = True) -> pd.DataFrame:
    g_index = pd.RangeIndex(len(df)) // n_iter
    df_ = df.groupby(g_index, as_index=False).cummin()
    df_['rs_iter'] = df.groupby(g_index).cumcount()
    df_['rs_group'] = g_index
    if append_index:
        df_.set_index(['rs_group', 'rs_iter'], append=True, inplace=True)
    return df_

# If this script is being run as the main script
if __name__ == '__main__':
    # Create a pseudorandom number generator with the specified seed
    random_ = np.random.RandomState(seed=SEED)

    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1])
    print(f"#Entries: {len(data)}")

    n_scene = ITER_COUNT * RS_REPEAT
    df = data.get(min_rep=EXP_REPEAT, max_rep=EXP_REPEAT, count=n_scene,
                  agg_mode=('min', 'mean'), random_state=SEED)
    
    rs_res = RS(df, n_iter=ITER_COUNT)
    rs_res = unstack_col_level(rs_res, 'agg_mode', level=0).reset_index()

    ylim_dict = dict(zip(fit_cols, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]))

    # Set the font scale for seaborn plots
    sns.set(font_scale=1.0)

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 5))
    for ax, col, label in zip(axes, fit_cols, fit_labels):
        # Create a line plot of the data for the fitness value
        sns.lineplot(x='rs_iter', y=col, hue='agg_mode', legend=False,
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
    diff = rs_res[rs_res.agg_mode == 'mean'] \
            .set_index(['rs_group', 'rs_iter'])[fit_cols] \
            .subtract(rs_res[rs_res.agg_mode == 'min'] \
            .set_index(['rs_group', 'rs_iter'])[fit_cols]) \
            .reset_index()
    
    # Add a box column to the data based on the index of the data point
    diff['box'] = diff['rs_iter'].apply(
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

    last_iter = rs_res.groupby(['rs_group', 'agg_mode', 'rs_iter']).last().reset_index()
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
        ax.set_ylim(*ylim_dict[col])

        ax.set_xticklabels(['RSwRep', 'RS'])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('end_box.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

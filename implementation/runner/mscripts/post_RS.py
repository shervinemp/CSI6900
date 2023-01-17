from functools import partial
from pprint import pprint
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from data_handler import get_values
from utils import CSVData, VD_A, in_cols, fit_cols, fit_labels

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
if __name__ == '__main__':
    # Create a pseudorandom number generator with the specified seed
    random_ = np.random.RandomState(seed=SEED)

    # Read in a list of experiments from a file specified as the first command line argument
    data = CSVData(sys.argv[1], min_run=EXP_REPEAT)
    print(f"#Entries: {len(data)}")

    exps = data.indices[random_.permutation(len(data))]
    ngroups = len(data)//ITER_COUNT if RS_REPEAT is None else RS_REPEAT

    # Divide the shuffled list of experiments into groups of size `ITER_COUNT` using list slicing
    s = [x*ITER_COUNT for x in range(ngroups+1)]
    df = data._df.groupby(in_cols).sample(EXP_REPEAT, random_state=random_)
    groups = pd.concat([df.loc[exps[beg:end]].assign(group_id=i) \
                        for i, (beg, end) in enumerate(zip(s, s[1:]))]) \
               .set_index('group_id', append=True)
    
    val_grp = groups.groupby(['group_id', *in_cols])[fit_cols]
    values_df = pd.concat([val_grp.min().assign(agg_mode='min'),
                           val_grp.mean().assign(agg_mode='mean')]) \
                  .groupby('group_id').sample(frac=1, random_state=random_)
    values_df['x'] = values_df.groupby(['group_id', 'agg_mode']).cumcount()
    
    res_grps = values_df.set_index(['x', 'agg_mode'], append=True) \
                        .groupby(['group_id', 'agg_mode']) \
                        .cummin() \
                        .reset_index()

    ylim_dict = dict(zip(fit_cols, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]))

    # Set the font scale for seaborn plots
    sns.set(font_scale=1.0)

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 5))
    for i, col in enumerate(fit_cols):
        ax = axes[i]

        # Create a line plot of the data for the fitness value
        g = sns.lineplot(x='x', y=col, hue='agg_mode', legend=False,
                         data=res_grps, ax=ax)
        
        ax.set_xlim((0, 50))
        ax.set_ylim(*ylim_dict[col])

        ax.set_xticks(range(0, 51, 10))

        # Set the x and y labels for the plot
        ax.set(xlabel='iteration', ylabel=fit_labels[i])

        ax.margins(0)
    # Set the legend labels
    fig.legend(labels=['RS', 'RSwRep'])
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
    diff = res_grps[res_grps.agg_mode == 'mean'].set_index(['group_id', 'x'])[fit_cols] \
           .subtract(res_grps[res_grps.agg_mode == 'min'].set_index(['group_id', 'x'])[fit_cols]) \
           .reset_index()
    
    # Add a box column to the data based on the index of the data point
    diff['box'] = diff['x'].apply(
        lambda x: next(i for i, b in enumerate(hist_bins) if x < b) - 1
    )

    # Iterate over the fitness values
    for i, col in enumerate(fit_cols):        
        ax = axes[i]
        
        # Create a box plot of the data
        sns.boxplot(x='box', y=col, data=diff, showmeans=True, ax=ax)
        
        # Set the x and y labels for the plot
        ax.set(xlabel='iteration', ylabel=fit_labels[i])
        
        # Set the x-axis tick labels
        ax.set_xticks([])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('rs_diff_plot.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

    last_iter = res_grps.groupby(['group_id', 'agg_mode', 'x']).last().reset_index()
    b_ = last_iter[last_iter.agg_mode == 'min']
    c_ = last_iter[last_iter.agg_mode == 'mean']
    
    df_end = pd.concat([b_, c_])

    print('avg min:')
    pprint(b_[fit_cols].mean())

    print('avg mean:')
    pprint(c_[fit_cols].mean())

    print("min-mean")
    pprint({l: wilcoxon(c_[col].to_list(), b_[col].to_list()) \
           for l, col in zip(fit_labels, fit_cols)})
    
    pprint({l: VD_A(c_[col].to_list(), b_[col].to_list()) \
           for l, col in zip(fit_labels, fit_cols)})
    
    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(4 * len(fit_cols), 4))

    # Iterate over the fitness values
    for i, col in enumerate(fit_cols):
        ax = axes[i]
        
        # Create a histogram of the data
        sns.boxplot(data=df_end, x='agg_mode', y=col, orient='v', showmeans=True, ax=ax)
        
        # Set the x and y labels for the plot
        ax.set(xlabel="aggregation", ylabel=fit_labels[i])

        ax.set_xticklabels(['RSwRep', 'RS'])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('end_box.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

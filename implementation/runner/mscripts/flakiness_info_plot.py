from pprint import pprint
from scipy.stats import wilcoxon
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_handler import get_values
from utils import CSVData, fit_cols, fit_labels, separate_suffix, neg_histplot

# Seed for the pseudorandom number generator
SEED = 0

# Number of experiments
COUNT = 1000

# Maximum number of experiment repeats
EXP_REPEAT = 10

# Read in a list of experiments from a file specified as the first command line argument
data = CSVData(sys.argv[1], min_run=EXP_REPEAT, max_run=EXP_REPEAT)
print(f"#Entries: {data.size}")

# TODO: sampling with replacement for different groups?

# Create a pseudorandom number generator with the specified seed
random_ = np.random.RandomState(seed=SEED)

# If this script is being run as the main script
if __name__ == '__main__':
    df = data._data

    mi = data.group_by_index().min()
    ma = data.group_by_index().max()
    mean = data.group_by_index().mean()
    first = data.group_by_index().first()

    var = data.group_by_index().var()
    delta = ma - mi

    hard_flaky = ((mi <= 0) & (ma > 0))[fit_cols].sum() / data.size

    t = list(zip((r:=[-1, 0., 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1]), r[1:]))
    delta_max = delta.max()
    delta_flaky = {(x1, x2): pd.concat([(d:=( (delta > x1 * delta_max) & \
                                              (delta <= x2 * delta_max) )[fit_cols].sum()),
                                        d / data.size], axis=1).rename(columns={0: 'count', 1: 'percent'}) \
                   for x1, x2 in t}
    
    # t2 = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    t2 = np.linspace(-3, 3, 7, endpoint=True) * df[fit_cols].std()
    
    # first_min = first.min()
    # first_delta_max = first.max() - first.min()
    df_min = df[fit_cols].min()
    df_std = df[fit_cols].std()
    df_mean = df[fit_cols].mean()

    df_delta_max = df[fit_cols].max() - df_min
    first_thresh = {x: pd.concat([(mean <= x + df_mean)[fit_cols].sum(),
                                  d / data.size], axis=1).rename(columns={0: 'count', 1: 'percent'}) \
                    for x in t2}
    
    # mi_min = mi.min()
    # mi_delta_max = mi.max() - mi.min()
    mi_thresh = {x: pd.concat([(mi <= x + df_mean)[fit_cols].sum(),
                               d / data.size], axis=1).rename(columns={0: 'count', 1: 'percent'}) \
                 for x in t2}

    print("Hard flaky:")
    pprint(hard_flaky)

    print("Delta flaky:")
    pprint(delta_flaky)

    print("First thresh:")
    pprint(first_thresh)

    print("Min thresh:")
    pprint(mi_thresh)
    # range_space = [np.linspace(-3 * df_std[f], 3 * df_std[f], 21, endpoint=True) for f in fit_cols]

    # first_thresh = {}
    # mi_thresh = {}

    # for rang, f in zip(range_space, fit_cols):
    #     t2 = list(zip(rang, rang[1:]))
    #     first_thresh[f] = [(d:=((first[f] > x) & (first[f] <= y)).sum()) / data.size \
    #                        for x, y in t2]
        
    #     mi_thresh[f] = [(d:=((mi[f] > x) & (mi[f] <= y)).sum()) / data.size \
    #                     for x, y in t2]

    # Set the font scale for seaborn plots
    sns.set(font_scale=1.0)
    sns.set_theme()  # <-- This actually changes the look of plots.

    bin_range = (-3, 1)
    bins = np.linspace(*bin_range, 16, endpoint=True)
    diff_ = (mean - mi) / df_std
    data_ = [ [((diff_[f] > a) & (diff_[f] <= b)).sum() \
              for a, b in zip(bins, bins[1:])] \
            for f in fit_cols]

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, 1, figsize=(8, 10))

    neg_histplot(data=data_, bin_range=bin_range, legend_labels=None, ax=axes)
    axes.set_ylabel('RS - RSwRep')

    axes.set_xlabel('Standard deviations around mean')

    fig.legend(labels=fit_labels)

    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('count_hist.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

    diff = (mean - mi) / df_std
    
    mean_melt = mean.melt(value_vars=fit_cols, var_name="fit_id",
                          value_name="mean_vals", ignore_index=False).set_index(["fit_id"], append=True)
    diff_melt = diff.melt(value_vars=fit_cols, var_name="fit_id",
                          value_name="delta_vals", ignore_index=False).set_index(["fit_id"], append=True)
    mean_diff = pd.merge(mean_melt, diff_melt,
                         left_index=True, right_index=True).reset_index().set_index(diff.index.names)

    ax = sns.lineplot(data=mean_diff.reset_index(drop=True),
                      x="mean_vals", y="delta_vals", hue="fit_id")
    ax.set_ylabel('diff')

    ax.set_xlabel('Standard deviations around mean')

    fig = ax.get_figure()
    fig.legend(labels=fit_labels)

    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('delta_hist.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

    metrics = pd.concat([delta, var], axis=1,
                        keys=['delta','var'], 
                        names=['metric', 'fitness'])

    # Get the column indices for the fitness values specified in the fit_cols variable
    fit_col_ids = [x - 1 for x in map(separate_suffix, fit_cols)]

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(2, len(fit_cols), figsize=(5 * len(fit_cols), 8))

    get_bins = lambda ma: [x for i, x in enumerate(np.linspace(0, ma, 21, endpoint=True)) \
                           if i in (0, 1, 2, 4, 7, 12, 20)]

    # Iterate over the fitness values
    for i, m in enumerate(['delta', 'var']):
        for j, col in enumerate(fit_cols):
            ax = axes[i][j]

            metric_max = metrics[m][col].max()
            
            # Create a histogram of the data
            sns.histplot(x=col, data=metrics[m], bins=get_bins(metric_max),
                         binrange=(0, metric_max), ax=ax)
            
            # Set the x and y labels for the plot
            ax.set(xlabel=f"{col} {m}", ylabel='count')

            ax.set_xticks(np.linspace(0, metric_max, 6, endpoint=True))

            ax.set_xlim([0, metric_max])
        
            # Set the x-axis tick labels
            ax.set_xticklabels(["{:.0f}%".format(x) for x in np.linspace(0, 100, 6, endpoint=True)])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('flaky_hist.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 4))

    df_list = []

    for i in range(1, 11):
        filtered = data.filter_max_run(i)
        min_df = filtered.groupby(level=filtered.index.names).min().assign(rep=i)
        df_list.append(min_df)

    df_ = pd.concat(df_list, axis=0, ignore_index=True)

    # pd.merge(df_list[0][fit_cols], df_list[1][fit_cols],
    #          left_index=True, right_index=True,
    #         suffixes=['_1rep', '_10rep']).to_csv('test.csv')

    pprint(df_.groupby('rep').mean()[fit_cols])
    pprint({f: wilcoxon((df_list[-1][f] - df_list[0][f]).to_list()) \
           for f in fit_cols})
    
    # Iterate over the fitness values
    for i, col in enumerate(fit_cols):
        ax = axes[i]
        
        # Create a histogram of the data
        sns.boxplot(data=df_[(df_.rep == 10) | (df_.rep == 1)],
                    x='rep', y=col, orient='v', showmeans=True, ax=ax)
        
        # Set the x and y labels for the plot
        ax.set(xlabel="repeats", ylabel=f"{col}")
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('flaky_box.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

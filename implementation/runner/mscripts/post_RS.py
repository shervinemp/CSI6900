from functools import partial
from pprint import pprint
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from data_handler import get_values
from utils import CSVData, VD_A, fit_cols, fit_labels, separate_suffix

# Seed for the pseudorandom number generator
SEED = 0

# Number of experiments in each group
ITER_COUNT = 50

# Maximum number of groups to create (if not None)
RS_REPEAT = 19

# Maximum number of experiment repeats
EXP_REPEAT = 10

HIST_SIZE = 25

# Read in a list of experiments from a file specified as the first command line argument
data = CSVData(sys.argv[1], min_run=EXP_REPEAT)
print(f"#Entries: {data.size}")

# TODO: sampling with replacement for different groups?

# Create a pseudorandom number generator with the specified seed
random_ = np.random.RandomState(seed=SEED)

def get_min_fitness_value(e, target_fit):
    # Get the list of fitness values for the experiments
    exp_f_values = data.get_f_values(e)[:EXP_REPEAT]
    # Return the minimum fitness value from the list of files
    return min(f_v[target_fit] for f_v in exp_f_values)

def get_mean_fitness_value(e, target_fit):
    # Get the list of fitness values for the experiments
    exp_f_values = data.get_f_values(e)[:EXP_REPEAT]
    # Return the minimum fitness value from the list of files
    return sum(f_v[target_fit] for f_v in exp_f_values) / len(exp_f_values)

# Create a function to get the first fitness value for a single experiment
def get_first_fitness_value(e, target_fit):
    # Get the first fitness values for the experiment
    exp_f_values = data.get_f_values(e)[0]
    # Return the first fitness value from the file
    return exp_f_values[target_fit]

# Define a function to compute the moving minimum of an iterable
def moving_minimum(iterable):
    # Yield the index and value of the minimum element in the iterable, as well as the index and value of the current element for each iteration
    min_value = np.inf
    for i, v in enumerate(iterable):
        if v < min_value:
            min_index, min_value = i, v
        yield min_index, min_value

# If this script is being run as the main script
if __name__ == '__main__':
    exps = list(data.indices)

    # Shuffle the list of experiments using the pseudorandom number generator
    random_.shuffle(exps)

    # Divide the shuffled list of experiments into groups of size `ITER_COUNT` using list slicing
    groups = [exps[i: i+ITER_COUNT] for i in range(0, len(exps) if RS_REPEAT is None else RS_REPEAT*ITER_COUNT, ITER_COUNT)]

    # Get the column indices for the fitness values specified in the fit_cols variable
    fit_col_ids = [x - 1 for x in map(separate_suffix, fit_cols)]

    # Create a list of dataframes containing the results for each group of experiments and fitness value
    res_grps = []
    for i, group in enumerate(groups):
        group = sorted(group, key=lambda k: random_.random())
        for fit_id in fit_col_ids:
            for agg_mode in ('mean', 'min'):
                # Get the minimum fitness values for the group of experiments
                if agg_mode == 'min':
                    fitness_values = moving_minimum(map(partial(get_min_fitness_value, target_fit=fit_id), group))
                elif agg_mode == 'first':
                    fitness_values = moving_minimum(map(partial(get_first_fitness_value, target_fit=fit_id), group))
                elif agg_mode == 'mean':
                    fitness_values = moving_minimum(map(partial(get_mean_fitness_value, target_fit=fit_id), group))

                # Create a dataframe with the results for the group of experiments and fitness value
                df = pd.DataFrame({'vals': [val[1] for val in fitness_values]})
                # Add the group, fit_id, and use_min columns to the dataframe
                df = df.reset_index().assign(grp=i, fit_id=fit_id, agg_mode=agg_mode)
                # Add the dataframe to the list of results
                res_grps.append(df)

    # Concatenate the list of dataframes into a single dataframe
    df = pd.concat(res_grps, axis=0, ignore_index=True)

    ylim_dict = dict(zip(fit_cols, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]))

    # Set the font scale for seaborn plots
    sns.set(font_scale=1.0)

    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(5 * len(fit_cols), 5))
    for i, (fid, col) in enumerate(zip(fit_col_ids, fit_cols)):
        ax = axes[i]

        # Create a line plot of the data for the fitness value
        g = sns.lineplot(x='index', y='vals', hue='agg_mode', legend=False,
                         data=df[df.fit_id == fid], ax=ax)
        
        ax.set_xlim((0, 50))
        ax.set_ylim(*ylim_dict[col])

        ax.set_xticks(range(0, 51, 10))

        # Set the x and y labels for the plot
        ax.set(xlabel='iteration', ylabel=fit_labels[i])

        ax.margins(0)

        # Set the legend labels
        # ax.legend(labels=['RS', 'RS-10rep', 'test'])
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
    hist_bins = np.linspace(0, len(groups[0]), HIST_SIZE + 1)

    # Iterate over the fitness values
    for i, (fid, col) in enumerate(zip(fit_col_ids, fit_cols)):
        # Create a copy of the data for the fitness value
        df_ = df[df.fit_id == fid].copy()
        
        ax = axes[i]
        
        # Get the data for mean fitness values
        df_nonmin = df_[df_.agg_mode == 'mean'].reset_index().drop('agg_mode', axis=1).copy()
        
        # Calculate the difference between non-minimum and minimum fitness values
        df_nonmin['vals'] = df_nonmin['vals'] - df_[df_.agg_mode == 'min'].reset_index()['vals']
        
        # Add a box column to the data based on the index of the data point
        df_nonmin['box'] = df_nonmin['index'].apply(
            lambda x: next(i for i, b in enumerate(hist_bins) if x < b) - 1
        )
        
        # Create a box plot of the data
        sns.boxplot(x='box', y='vals', data=df_nonmin, showmeans=True, ax=ax)
        
        # Set the x and y labels for the plot
        ax.set(xlabel='iteration', ylabel=fit_labels[i])
        
        # Set the x-axis tick labels
        ax.set_xticks([])
        # ax.set_xticklabels([str(x)[:4] for x in hist_bins])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('rs_diff_plot.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

    last_iter = df[df['index'] == ITER_COUNT-1]
    # a_ = last_iter[last_iter.agg_mode == 'first']
    b_ = last_iter[last_iter.agg_mode == 'min']
    c_ = last_iter[last_iter.agg_mode == 'mean']
    
    df_end = pd.concat([b_, c_], axis=0, ignore_index=True)

    # print('avg first:')
    # pprint(list(zip(fit_labels, a_.groupby("fit_id").mean()['vals'])))

    print('avg min:')
    pprint(list(zip(fit_labels, b_.groupby("fit_id").mean()['vals'])))

    print('avg mean:')
    pprint(list(zip(fit_labels, c_.groupby("fit_id").mean()['vals'])))

    # print("min-first")
    # pprint({l: wilcoxon((a_[a_.fit_id == fid].set_index('grp')['vals'] - b_[b_.fit_id == fid].set_index('grp')['vals']).to_list()) \
    #        for l, fid in zip(fit_labels, fit_col_ids)})
    
    # pprint({l: VD_A((a_[a_.fit_id == fid].set_index('grp')['vals']).to_list(),
    #                 (b_[b_.fit_id == fid].set_index('grp')['vals']).to_list()) \
    #        for l, fid in zip(fit_labels, fit_col_ids)})

    print("min-mean")
    pprint({l: wilcoxon((c_[c_.fit_id == fid].set_index('grp')['vals'] - b_[b_.fit_id == fid].set_index('grp')['vals']).to_list()) \
           for l, fid in zip(fit_labels, fit_col_ids)})
    
    pprint({l: VD_A((c_[c_.fit_id == fid].set_index('grp')['vals']).to_list(),
                    (b_[b_.fit_id == fid].set_index('grp')['vals']).to_list()) \
           for l, fid in zip(fit_labels, fit_col_ids)})
    
    # Create a subplot with one plot for each fitness value
    fig, axes = plt.subplots(1, len(fit_cols), figsize=(4 * len(fit_cols), 4))

    # Iterate over the fitness values
    for i, col in enumerate(fit_cols):
        ax = axes[i]
        
        # Create a histogram of the data
        sns.boxplot(data=df_end[df_end.fit_id == separate_suffix(col)-1],
                    x='agg_mode', y='vals', orient='v', showmeans=True, ax=ax)
        
        # Set the x and y labels for the plot
        ax.set(xlabel="aggregation", ylabel=fit_labels[i])

        ax.set_xticklabels(['RSwRep', 'RS'])
    # Tightly adjust the layout of the plots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot to a file
    plt.savefig('end_box.pdf', bbox_inches='tight')
    # Close the plot
    plt.close()

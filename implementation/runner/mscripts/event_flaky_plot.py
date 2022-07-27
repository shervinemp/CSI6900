import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from measures import *


if __name__ == '__main__':
    whole_df = pd.read_csv(sys.argv[1], index_col=0)
    ind_dfs = []
    ind_cols = [f'event{i}' for i in ['1', '1_1', '1_2', '1_3', '1_4', '2_1', '2_2', '3']] + ['idx']
    cum_dfs = []
    cum_cols = ['event4', 'event5', 'event6']
    whole_df.iloc[:, 12] *= 10
    run_thresh = 10
    csv_files = map(lambda x: str(list(x))+'.csv', whole_df.groupby(whole_df.columns[:16].to_list()).size().index)
    for idx, f in enumerate(csv_files):
        input_df = pd.read_csv(f, index_col=0)
        runs = input_df['run'].max() + 1
        if runs < run_thresh:
            continue;
        for i in range(runs):
            arr = [
                event1(input_df, i),
                event1_1(input_df, i),
                event1_2(input_df, i),
                event1_3(input_df, i),
                event1_4(input_df, i),
                event2_1(input_df, i),
                event2_2(input_df, i),
                event3(input_df, i),
                idx
            ]
            ind_dfs.append(arr)
        arr = [
            event4(input_df),
            event5(input_df),
            event6(input_df)
        ]
        cum_dfs.append(arr)
    ind_df = pd.DataFrame(ind_dfs, columns=ind_cols).groupby('idx').mean()
    ind_df.index = list(map(int, ind_df.index))
    cum_df = pd.DataFrame(cum_dfs, columns=cum_cols)
    event_df = pd.merge(ind_df, cum_df, left_index=True, right_index=True)
    flaky_df = pd.merge(soft_flaky(whole_df), hard_flaky(whole_df), suffixes=('_soft', '_hard'), left_index=True, right_index=True)
    measure_df = pd.merge(event_df, flaky_df, left_index=True, right_index=True)

    rows, cols = len(flaky_df.columns), len(event_df.columns)
    fig, axes = plt.subplots(rows, cols, figsize=(40, 40))
    plt.setp(axes, ylim=(0., 1.01), yticks=[])
    for i in range(rows):
        for j in range(cols):
            x = event_df.iloc[:, j]
            order = x.sort_values().index
            y = flaky_df.iloc[:, i]
            ax = axes[i][j]
            ax.scatter(x[order], y[order])
            ax.set_title('e{}_{}'.format(event_df.columns[j][5:], flaky_df.columns[i]))
    fig.savefig('plots.png')
    measure_df.to_csv('measure.csv')
    corr = measure_df.corr().dropna(axis=0, how='all').dropna(axis=1, how='all')
    hm = sns.heatmap(corr.iloc[11:, :11], cmap="viridis")
    hm.get_figure().savefig('corr.png')
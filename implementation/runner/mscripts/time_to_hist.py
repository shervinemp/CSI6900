import re
import sys
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

regex = re.compile("\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} \| Location\(x=[^,\n]+, y=[^,\n]+, z=[^,\n]+\)>DE:([^,\n]+),DfC:([^,\n]+),DfV:([^,\n]+),DfP:([^,\n]+),DfM:([^,\n]+),DT:([^,\n]+),DiSO:([^,\n]*),DiLS:([^,\n]*),DiRI:([^,\n]*),DiID:([^,\n]*)")
columns = ['DE', 'DfC', 'DfV', 'DfP', 'DfM', 'DT', 'DiSO', 'DiLS', 'DiRI', 'DiID']

def make_float(v):
    try:
        return float(v)
    except Exception:
        return str(v)

def prod_pair(arr):
    split = arr.split('-')
    marr = []
    for i in range(len(split)):
        if len(marr):
            if len(marr[-1]):
                if marr[-1][-1] == 'e':
                    marr[-1] = marr[-1] + split[i]
            else:
                marr[-1] = split[i]
        else:
            marr.append(split[i])
    if len(marr[-1]) == 0:
        marr = marr[:-1]
    split_arr = np.array([x.split('_') for x in marr])
    if len(split_arr) == 0:
        return []
    if split_arr.shape[1] == 2:
        return {int(a): make_float(b) for a, b in split_arr}
    else:
        return split_arr[:, 0].astype(float)

LINE_PLOT = True

if __name__ == '__main__':
    if sys.argv[1].split('.')[-1] == 'csv':
        time_df = pd.read_csv(sys.argv[1], index_col=0)
        time_df = time_df.drop(time_df.columns.difference([*columns, 'run']), axis=1)
    # else:
    #     with open(sys.argv[1], 'rt') as f:
    #         time_df = pd.DataFrame([[(prod_pair(x) if i >= columns.index('DiSO') else float(x)) \
    #                                 for i, x in enumerate(e)] \
    #                                 for e in regex.findall(f.read())], columns=columns)
    time_df['DE'] = (time_df.set_index('run')['DE'] - time_df.groupby('run').first()['DE']).to_list()
    time_ord_df = time_df.drop('DE', axis=1).iloc[:, :columns.index('DT')].unstack().to_frame().reset_index(0).join(time_df[['DE', 'run']]).rename(columns={'level_0': 'col', 0: 'val'}).reset_index(drop=False)
    for col in columns[1:columns.index('DT')+1]:
        time_ord_col = time_ord_df[time_ord_df['col'] == col]
        time_hist = time_ord_col.copy(deep=False)
        fig, ax = plt.subplots(figsize=(25,10))
        # rmi, rma = time_hist[time_hist['val'] < 950]['val'].min(), time_hist[time_hist['val'] < 950]['val'].max()
        rmi, rma = time_hist['val'].min(), time_hist['val'].max()
        if LINE_PLOT is True:
            sns.lineplot(x='DE', y='val', ci=None,
                        hue='run', data=time_hist)
        # else:
        #     mi, ma = time_hist['DE'].min(), time_hist['DE'].max()
        #     hists = np.linspace(mi, ma, 25, endpoint=False)
        #     time_hist['DE'] = time_hist['DE'].apply(lambda x: ((hists > x).tolist() + [True]).index(1) - 1)
        #     sns.boxplot(x='DE', y='val',
        #                 hue='col', data=time_hist)
        #     ax.set_xticklabels(list(map(lambda x: str(x)[:4], hists)))
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        ax.set(ylim=(rmi-.01, rma+.01))
        ax.legend(loc='upper left')
        sns.despine(left=True, bottom=True)

        fig.savefig('/'.join(sys.argv[1].split('/')[:-1]) + f'/{col}_plot.jpg')
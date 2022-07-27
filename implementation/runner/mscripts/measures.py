import pandas as pd
import numpy as np
from scipy.stats import entropy

def event1(input_df, run=None):
    eps = 1
    if run is None:
        df = input_df
    else:
        df = input_df[input_df.run == run]
    if df.DiLS.hasnans:
        return 0
    DT_end = df.DT.iat[len(df)-1]
    nomov = df[df.DT - DT_end < eps]
    light = nomov.DiLS.map(lambda x: x.split('-')[1].split('_')[1])
    event = (light == 'Green')
    return event.sum()

def event1_1(input_df, run=None):
    eps = 1
    if run is None:
        df = input_df
    else:
        df = input_df[input_df.run == run]
    if df.DiLS.hasnans:
        return 0
    DT_end = df.DT.iat[len(df)-1]
    nomov = df[df.DT - DT_end < eps]
    light = nomov.DiLS.map(lambda x: x.split('-')[1].split('_')[1])
    TG = ((light == 'Green').astype(int).diff() == 1)
    event = light[TG.cumsum() > 0]
    return len(event)

def event1_2(input_df, run=None):
    eps = 1
    psi = 20
    if run is None:
        df = input_df
    else:
        df = input_df[input_df.run == run]
    if df.DiLS.hasnans:
        return 0
    DT_end = df.DT.iat[len(df)-1]
    nomov = df[df.DT - DT_end < eps]
    bv = nomov[nomov.DfV < psi]
    indices = []
    sigma = 2
    ma = 0
    for i, e in enumerate(bv.DfV.values[::-1]):
        if ma - e > sigma:
            indices.append(i)
        if e > ma:
            ma = e
    indices = bv.index[::-1][indices][::-1]
    if len(indices) == 0:
        return 0
    event = (df.loc[indices[0]:].DiLS.map(lambda x: x.split('-')[1].split('_')[1]) == 'Green')
    return event.sum()

def event1_3(input_df, run=None):
    eps = 1
    k = 25
    if run is None:
        df = input_df
    else:
        df = input_df[input_df.run == run]
    DT_end = df.DT.iat[len(df)-1]
    nomov = df[df.DT - DT_end < eps]
    max_std = 0
    for i in nomov.index:
        kstd = df.DfC[max(0, i-k): i+k].std()
        if kstd > max_std:
            max_std = kstd
    return max_std

def event1_4(input_df, run=None):
    eps = 1
    if run is None:
        df = input_df
    else:
        df = input_df[input_df.run == run]
    DT_end = df.DT.iat[len(df)-1]
    nomov = df[df.DT - DT_end < eps]
    return len(nomov)

def event2_1(input_df, run=None):
    psi = 20
    k = 25
    if run is None:
        df = input_df
    else:
        df = input_df[input_df.run == run]
    near = df[df.DfV < psi]
    max_std = 0
    for i in near.index:
        kstd = df.DfC[max(0, i-k): i+k].std()
        if kstd > max_std:
            max_std = kstd
    return max_std

def event2_2(input_df, run=None):
    psi = 20
    k = 25
    if run is None:
        df = input_df
    else:
        df = input_df[input_df.run == run]
    max_std = 0
    for i in df.index:
        kstd = df.DfC[max(0, i-k): i+k].std()
        if kstd > max_std:
            max_std = kstd
    return max_std

def event3(input_df, run=None):
    psi = 20
    k = 25
    if run is None:
        df = input_df
    else:
        df = input_df[input_df.run == run]
    accel = df.DT.diff().diff().fillna(0)
    max_std = 0
    for i in df.index:
        kstd = accel[max(0, i-k): i+k].std()
        if kstd > max_std:
            max_std = kstd
    return max_std

def event4(input_df):
    roads = input_df.groupby('run').last().DiRI
    event = (roads == roads.iat[0])
    return event.all()

def event5(input_df):
    event = input_df.groupby('run').first().DE
    return event.std()

def event6(input_df):
    if input_df.DiLS.hasnans:
        return 0
    edge = input_df.DiLS.map(lambda x: x.split('-')[1].split('_')[1]).map({'Red': 0, 'Yellow': 1, 'Green': 2}).diff().fillna(0) != 0
    run_len = input_df.groupby('run')['DE'].count()
    light_change = [[e for e in x if e != 0] for x in input_df[edge].groupby('run').groups.values()]
    max_len = max(map(len, light_change))
    padded = [(x + [m for i in range(max_len - len(x))]) for x, m in zip(light_change, run_len)]
    events = np.array(padded)
    return events.std(axis=0).mean()

def soft_flaky(whole_df):
    fit_cols = [f'f{i}' for i in range(6)]
    normalized = whole_df.copy()
    normalized[fit_cols] = whole_df[fit_cols] / 1
    return 4 * normalized.groupby('case')[fit_cols].var()

def hard_flaky(whole_df):
    fit_cols = [f'f{i}' for i in range(6)]
    normalized = whole_df.copy()
    normalized[fit_cols] = whole_df[fit_cols] / 1
    hard_df = normalized.copy()
    hard_df[fit_cols] = hard_df[fit_cols] > 0
    return hard_df.groupby('case')[fit_cols].agg(lambda x: entropy([x.sum(), x.count() - x.sum()], base=2))
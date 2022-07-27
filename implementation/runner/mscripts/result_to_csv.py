from fcntl import F_SEAL_SHRINK
import re
import sys
import pandas as pd
from utils import fit_cols, val_cols

regex = re.compile("\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[([\d\s,]+)\]:((?:[+-]?(?:[0-9]*[.])?[0-9]+,?)+)")

arrs = [
    [2, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 3, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 3, 0, 4, 1, 1, 0],
    [3, 3, 0, 0, 1, 0, 0, 0, 0, 1, 3, 0, 3, 1, 1, 2],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 2, 1, 0, 2, 1, 1, 0],
    [3, 2, 0, 1, 1, 0, 1, 1, 1, 2, 5, 0, 3, 1, 1, 0],
    [3, 1, 0, 1, 1, 1, 1, 1, 0, 2, 1, 0, 3, 1, 1, 1],
    [2, 0, 0, 0, 1, 0, 1, 0, 1, 0, 3, 0, 2, 1, 1, 0],
    [1, 1, 0, 1, 0, 0, 1, 1, 0, 2, 2, 0, 2, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 1, 4, 1, 1, 0],
    [3, 1, 0, 1, 0, 1, 1, 0, 1, 1, 2, 0, 3, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1, 0, 2, 1, 1, 0],
    [3, 1, 0, 1, 0, 0, 1, 0, 1, 2, 2, 1, 3, 1, 1, 1],
    [3, 1, 0, 1, 0, 1, 1, 0, 0, 0, 5, 1, 4, 1, 1, 1],
    [3, 1, 0, 1, 0, 0, 1, 0, 1, 1, 3, 0, 3, 1, 1, 0],
    [3, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 3, 1, 1, 2],
    [1, 2, 0, 1, 1, 1, 1, 0, 0, 0, 3, 0, 4, 1, 1, 0],
    [3, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 4, 1, 1, 1],
    [3, 1, 0, 1, 1, 0, 0, 0, 1, 2, 6, 1, 4, 1, 1, 0],
    [3, 2, 0, 1, 1, 1, 1, 0, 1, 1, 6, 0, 4, 1, 1, 2],
    [3, 3, 0, 1, 1, 0, 1, 0, 0, 2, 2, 1, 2, 1, 1, 0],
    [2, 3, 0, 1, 1, 0, 0, 0, 1, 2, 0, 1, 4, 1, 1, 0],
    [2, 3, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 4, 1, 1, 0],
    [2, 3, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 4, 1, 1, 0],
    [2, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 1, 1, 0],
    [2, 3, 0, 0, 0, 1, 1, 1, 0, 1, 4, 0, 2, 1, 1, 0],
    [2, 3, 0, 0, 0, 0, 1, 1, 1, 2, 4, 0, 2, 1, 1, 0],
    [2, 3, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 4, 1, 1, 0],
    [2, 3, 0, 0, 1, 0, 1, 1, 1, 1, 3, 0, 4, 1, 1, 0],
    [2, 3, 0, 1, 0, 0, 0, 1, 1, 1, 3, 0, 2, 1, 1, 0],
    [2, 3, 0, 0, 1, 0, 0, 0, 1, 1, 4, 0, 4, 1, 1, 0],
    [3, 2, 0, 0, 0, 0, 1, 1, 1, 1, 4, 0, 3, 1, 1, 1],
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 2, 0, 1, 2, 1, 1, 0],
    [1, 2, 0, 0, 1, 0, 0, 0, 1, 2, 6, 0, 3, 1, 1, 0],
    [3, 3, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 2, 1, 1, 0],
    [3, 3, 0, 1, 0, 1, 1, 0, 1, 0, 3, 1, 2, 1, 1, 0],
    [3, 2, 0, 1, 0, 0, 0, 1, 0, 1, 6, 1, 3, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 2, 0, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 4, 1, 3, 1, 1, 0],
    [3, 3, 0, 1, 1, 0, 1, 1, 1, 2, 0, 0, 2, 1, 1, 1],
    [3, 2, 0, 0, 0, 0, 1, 0, 0, 2, 6, 1, 2, 1, 1, 1],
]

arrs = None

AUTO_GROUP = True

def create_dict(V, F):
    d = {}
    vect = list(map(int, V.split(',')))
    fitn = list(map(float, F.split(',')))
    for c, v in zip(val_cols, vect):
        d[c] = v
    for c, f in zip(fit_cols, fitn):
        d[c] = f
    return d

def parse_file(addr):
    with open(addr, 'rt') as f:
        df = pd.DataFrame([create_dict(V, F) for V, F in regex.findall(f.read())])
    return df

def mlookup(mcase):
    for i, case2 in enumerate(map(tuple, arrs)):
        if mcase == case2:
            return i

def get_case(x):
    a = x.iloc[:16].astype(int).to_list()
    a[-4] = a[-4] // 10
    return tuple(a)

if __name__ == '__main__':
    df = pd.concat([parse_file(addr).assign(run=i) for i, addr in enumerate(sys.argv[1:])], axis=0)
    if AUTO_GROUP is True:
        df['run'] = df.groupby(val_cols).cumcount()
    if arrs is not None:
        df['case'] = df.apply(lambda x: mlookup(get_case(x)), axis=1)
    get_dir = lambda x: '/'.join(x.split('/')[:-1]) + '/'
    is_same_folder = True
    def_path = get_dir(sys.argv[1])
    for p in sys.argv[2:]:
        if get_dir(p) != def_path:
            is_same_folder = False
            break
    df.to_csv(('.'.join(sys.argv[1].split('.')[:-1]) if len(sys.argv) == 2 \
        else ((def_path if is_same_folder else '') + 'compiled_res')) + '.csv')

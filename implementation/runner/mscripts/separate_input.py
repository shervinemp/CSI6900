import pandas as pd
import sys
from math import isnan
from utils import fit_cols

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], index_col=0)
    mdf = df.reset_index().set_index(['index', 'run']).unstack()
    # mdf.to_csv(f"{'.'.join(sys.argv[1].split('.')[:-1])}_sep.csv")
    for col in fit_cols:
        mdf[col].to_csv(f"{'.'.join(sys.argv[1].split('.')[:-1])}_{col}.csv", index=False)
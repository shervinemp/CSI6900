import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from rs import RandomSearch as RS
import seaborn as sns

from data import CSVDataLoader, fit_cols, fit_labels
from stat_utils import stat_test
from utils import unstack_col_level

# Seed for the pseudorandom number generator
SEED = 0
# Number of experiments in each group
ITER_COUNT = 50
HIST_SIZE = 25

# If this script is being run as the main script
if __name__ == "__main__":
    # Create a pseudorandom number generator with the specified seed
    random_ = np.random.RandomState(seed=SEED)

    # Read in a list of experiments from a file specified as the first command line argument
    df = CSVDataLoader(sys.argv[1]).get(agg_mode=("min", "mean"))

    rs_res = RS.from_dataframe(df, n_iter=ITER_COUNT)
    rs_res = unstack_col_level(rs_res, "agg_mode", level=0)

    ylim_dict = dict(zip(fit_cols, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]))

    rs_res.plot_rs("agg_mode", ylim_dict=ylim_dict, legend_kwargs=dict(labels=["RSwRep", "RS"]))

    # Calculate the difference between non-minimum and minimum fitness values
    diff = (
        rs_res.query("agg_mode == \"mean\"")[fit_cols]
        .subtract(
            rs_res.query("agg_mode == \"min\"")[fit_cols]
        )
    )

    diff.plot_rs_box(HIST_SIZE)

    last_iter = rs_res.get_last_iter(groupby="agg_mode")
    l_iter_min = last_iter[last_iter.agg_mode == "min"][fit_cols]
    l_iter_mean = last_iter[last_iter.agg_mode == "mean"][fit_cols]

    print("avg min:")
    pprint(l_iter_min.mean())

    print("avg mean:")
    pprint(l_iter_mean.mean())

    print("min-mean")
    stat_test(l_iter_min, l_iter_mean)

    rs_res.plot_converge_box("agg_mode", output_file="endbox.pdf", ylim_dict=ylim_dict)

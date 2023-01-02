from glob import glob
from collections import Counter
import numpy as np
import sys

if __name__ == '__main__':
    files = glob('[[]*')
    files = list(filter(lambda x: x.split('.')[-1] not in ['csv', 'ogv', 'log'], files))
    files_ = list(map(lambda x: x.split('_')[0], files))
    cnt = Counter(files_)
    lower, upper = 0, float('inf')
    if len(sys.argv) > 1:
        lower = sys.argv[1]
        if len(sys.argv) == 3:
            upper = sys.argv[2]
        else:
            raise ValueError('Too many arguments.')
    idx = [(x>=int(sys.argv[1]) and x<int(sys.argv[2])) for x in cnt.values()]
    mfi = np.array(list(cnt.keys()))[idx]
    mfi_ = list(map(eval, mfi))
    mfi_ = [(x[:-4] + [x[-4]//10] + x[-3:]) for x in mfi_]
    with open(f'mfiles+{lower}-{upper}.txt', 'wt') as f:
        f.write(str(mfi_))

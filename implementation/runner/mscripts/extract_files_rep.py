from glob import glob
from collections import Counter
import numpy as np
import sys

if __name__ == '__main__':
    files = glob('[[]*')
    files = list(filter(lambda x: x.split('.')[-1] not in ['csv', 'ogv', 'log'], files))
    files_ = list(map(lambda x: x.split('_')[0], files))
    cnt = Counter(files_)
    if len(sys.argv) == 2:
        idx = [(x>=int(sys.argv[1])) for x in cnt.values()]
    elif len(sys.argv) == 3:
        idx = [(x>=int(sys.argv[1]) and x<int(sys.argv[2])) for x in cnt.values()]
    mfi = np.array(list(cnt.keys()))[idx]
    mfi_ = list(map(eval, mfi))
    mfi_ = [(x[:-4] + [x[-4]//10] + x[-3:]) for x in mfi_]
    with open('mfiles+10.txt', 'wt') as f:
        f.write(str(mfi_))

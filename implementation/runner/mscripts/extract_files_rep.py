from collections import Counter
import numpy as np
import sys

if __name__ == '__main__':
    files = glob('[*')
    files = list(filter(lambda x: x.split('.')[-1]!='csv', files))
    files_ = list(map(lambda x: x.split('_')[0], files))
    cnt = Counter(files_)
    if len(sys.argv) == 1:
        idx = [(x>=sys.argv[1]) for x in cnt.values()]
    elif len(sys.argv) == 2:
        idx = [(x>=sys.argv[1] and x<sys.argv[2]) for x in cnt.values()]
    mfi = np.array(list(cnt.keys()))[a]
    mfi_ = list(map(lambda x: x[:38]+x[39:], mfi))
    with open('mfiles+10.txt', 'wt') as f:
        f.write(str(mfi_))

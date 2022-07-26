import re
import os
import sys
import pandas as pd

regex = re.compile("\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2} \| Location\(x=([^,\n]+), y=([^,\n]+), z=([^,\n]+)\)>DE:([^,\n]+),DfC:([^,\n]+),DfV:([^,\n]+),DfP:([^,\n]+),DfM:([^,\n]+),DT:([^,\n]+),DiSO:([^,\n]*),DiLS:([^,\n]*),DiRI:([^,\n]*),DiID:([^,\n]*)")

columns = ['x', 'y', 'z', 'DE', 'DfC', 'DfV', 'DfP', 'DfM', 'DT', 'DiSO', 'DiLS', 'DiRI', 'DiID']

def parse_file(addr):
    with open(addr, 'rt', errors='ignore') as f:
        df = pd.DataFrame(regex.findall(f.read()), columns=columns)
    return df

if __name__ == '__main__':
    df = pd.concat([parse_file(addr).assign(run=i) for i, addr in enumerate(sys.argv[1:])], axis=0)

    get_dir = lambda x: x.split('/')[-1].split('_')[0]
    is_same_folder = True
    def_path = get_dir(sys.argv[1])
    for p in sys.argv[2:]:
        if get_dir(p) != def_path:
            is_same_folder = False
            break
    df.to_csv((sys.argv[1] if len(sys.argv) == 2 \
        else (def_path if is_same_folder else 'compiled_time')) + '.csv')

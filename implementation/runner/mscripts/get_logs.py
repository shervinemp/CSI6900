from glob import glob
from collections import defaultdict
import sys


if __name__ == "__main__":
    mfiles = glob("[*")
    grouped = False
    grouponly = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--grouped":
            grouped = True
        elif sys.argv[1] == "--grouponly":
            grouped = True
            grouponly = True
    filtered = defaultdict(list)
    for e in mfiles:
        if e[-4:] not in (".ogv", ".csv"):
            if grouped:
                filtered[e.split("/")[-1].split("_")[0]].append(e)
            else:
                filtered[e].append(e)
    if grouponly:
        for e in filtered.keys():
            print('"' + e + '"')
    else:
        for names in filtered.values():
            print('"' + '" "'.join(names) + '"')

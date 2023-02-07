from data_handler import get_values
from glob import glob


if __name__ == "__main__":
    mfiles = glob("[[]*")
    with open("extracted_fitnesses.log", "wt") as out_file:
        for e in mfiles:
            if e[-4:] not in (".ogv", ".log", ".csv"):
                fv, time = e.split("/")[-1].split("_")
                fv = eval(fv)
                fv[-4] //= 10
                time = (
                    time[:-13]
                    + ":"
                    + time[-12:-10]
                    + ":"
                    + time[-9:-7]
                    + ","
                    + time[-6:]
                )
                time = time[:-3]
                vals = get_values(e)
                out_file.write(
                    str(time) + " " + str(fv) + ":" + ",".join(map(str, vals)) + "\n"
                )

from glob import glob
from datetime import datetime


def total_system_time(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        first_line = lines[0]
        last_line = lines[-1]
        first_date = datetime.strptime(first_line[:19], "%m-%d-%Y %H:%M:%S")
        last_date = datetime.strptime(last_line[:19], "%m-%d-%Y %H:%M:%S")
        return (last_date - first_date).total_seconds()


if __name__ == "__main__":
    mfiles = glob("[[]*")
    system_times = [
        total_system_time(e) for e in mfiles if e[-4:] not in (".ogv", ".log", ".csv")
    ]
    print(f"Avg system sim time: {sum(system_times) / len(system_times)}")

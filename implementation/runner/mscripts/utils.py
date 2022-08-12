from glob import glob

val_cols = [
        "Road type",
        "Road ID",
        "Scenario Length", 
        "Vehicle_in_front", 
        "vehicle_in_adjcent_lane", 
        "vehicle_in_opposite_lane", 
        "vehicle_in_front_two_wheeled", 
        "vehicle_in_adjacent_two_wheeled", 
        "vehicle_in_opposite_two_wheeled",
        "time of day", 
        "weather", 
        "Number of People", 
        "Target Speed", 
        "Trees in scenario", 
        "Buildings in Scenario", 
        "task"
]

# fit_cols = ['DE', 'DfC', 'DfV', 'DfP', 'DfM', 'DT']
fit_cols = [f'f{i+1}' for i in range(6)]

def get_fv_files(fv):
        fv_ = [x for x in fv]
        fv_[-4] = fv_[-4] * 10
        return list(filter(lambda x: (x[-4] != '.' and x[-5] != '.'),
                           glob(f'[[]{", ".join(map(str, fv_))}[]]*')))
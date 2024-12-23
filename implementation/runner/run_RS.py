from itertools import count
import sys
sys.path.append('lib/')
import multiprocessing
from datetime import datetime
import time
import logging
from runner import run_single_scenario
from RS import *

RECORD = False
MAX_ITERS = 20
REPEATS = 10
TOTAL_TIME_BUDGET = None

class Pylot_caseStudy():
    def __init__(self):
        logger = logging.getLogger('logger')

        now = datetime.now()
        log_file = 'output/temp/' + str(i) + 'RS' + str(now) + '.log'
        logging.basicConfig(filename=log_file,
                            format='%(asctime)s %(message)s')

        logger.setLevel(logging.INFO)
        logger.info("Started")
    def _evaluate(self,x):
        fv = x
        if fv[0] != 3:
            fv[15] = 0
        DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = run_single_scenario(fv, record=RECORD)

        logger = logging.getLogger('logger')
        logger.info(str(fv)+':'+ str(DfC_min)+ ','+str( DfV_max)+','+str(DfP_max)+ ','+str(DfM_max)+ ','+str( DT_max) +','+ str(traffic_lights_max))

        return [DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max]



def run(i,archive):
    size = 1
    lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0]
    ub = [4, 4, 0, 2, 2, 2, 2, 2, 2, 3, 7, 2, 5, 2, 2, 3]
    # 0 Road type
    # 1 Road ID
    # 2 Scenario Length
    # 3 Vehicle_in_front
    # 4 vehicle_in_adjcent_lane
    # 5 vehicle_in_opposite_lane
    # 6 vehicle_in_front_two_wheeled
    # 7 vehicle_in_adjacent_two_wheeled
    # 8 vehicle_in_opposite_two_wheeled
    # 9 time of day
    # 10 weather
    # 11 Number of People
    # 12 Target Speed
    # 13 Trees in scenario
    # 14 Buildings in Scenario
    # 15 task
    threshold_criteria = [0,0,0,0,0.95,0]

    time_budget = 7200  # second
    no_of_Objectives = 6;


    now = datetime.now()
    logger = logging.getLogger()
    log_file = 'output/temp/' + str(i) + 'RS' + str(now) + '.log'
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s')

    logger.setLevel(logging.INFO)

    archive = minimize(Pylot_caseStudy()._evaluate, size, lb, ub, no_of_Objectives, threshold_criteria, time_budget, logger, archive, max_iters=MAX_ITERS, repeats=REPEATS)
    logger.info("Iteration completed")


if __name__ == "__main__":
    #
    print("in main")
    times_of_repetitions = 10
    for i in range(0, times_of_repetitions):
        manager = multiprocessing.Manager()
        archive = manager.list()
        p = multiprocessing.Process(target=run, name="run", args=(i,archive,))

        p.start()

        for t in (count(start=0, step=1) if TOTAL_TIME_BUDGET is None else range(TOTAL_TIME_BUDGET)):

            if p.is_alive():
                time.sleep(60)
            else:
                break

        p.terminate()

        # Cleanup
        p.join()

import sys
sys.path.append('lib/')
import multiprocessing
from datetime import datetime
import time
import logging
from runner import run_single_scenario
import numpy as np

class Pylot_caseStudy():
    def __init__(self):
        logger = logging.getLogger()

        now = datetime.now()
        log_file = 'output/temp/' + str(i) + 'avg' + str(now) + '.log'
        logging.basicConfig(filename=log_file,
                            format='%(asctime)s %(message)s')

        logger.setLevel(logging.DEBUG)
        logger.info("Started")
    def _evaluate(self,x):
        fv = x
        if fv[0] != 3:
            fv[15] = 0
        DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max = run_single_scenario(fv)

        logger = logging.getLogger()
        logger.info(str(fv)+':'+ str(DfC_min)+ ','+str(DfV_max)+','+str(DfP_max)+ ','+str(DfM_max)+ ','+str( DT_max) +','+ str(traffic_lights_max))

        return [DfC_min, DfV_max, DfP_max, DfM_max, DT_max, traffic_lights_max]



def run(exp, n):
    size = 6
    lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
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

    time_budget = 660  # second
    no_of_Objectives = 6;


    now = datetime.now()
    logger = logging.getLogger()
    log_file = 'output/temp/' + 'avg' + str(exp) + '_' + str(now) + '.log'
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s')

    logger.setLevel(logging.DEBUG)

    case = Pylot_caseStudy()
    vals = [case._evaluate(exp) for i in range(n)]
    avg = np.mean(vals, axis=0)
    std = np.std(vals, axis=0)
    logger.info("avg: " + str(avg) + ", std: " + str(std))
    logger.info("Iteration completed")


if __name__ == "__main__":
    #
    print("in main")
    times_of_repetitions = 1
    # exp = list(map(int, sys.argv[1].split(',')))
    # exp = [3, 1, 0, 1, 1, 1, 1, 0, 0, 2, 3, 0, 4, 1, 1, 2]
    # exp = [1, 3, 0, 1, 0, 1, 0, 1, 0, 0, 6, 0, 3, 1, 1, 0]
    exp = [3, 0, 0, 1, 1, 1, 0, 1, 0, 2, 3, 1, 4, 1, 1, 2]
    for i in range(0, times_of_repetitions):
        p = multiprocessing.Process(target=run, name="run", args=(exp, 8))

        p.start()

        for t in range(60):

            if p.is_alive():
                time.sleep(60)
            else:
                break

        p.terminate()

        # Cleanup
        p.join()

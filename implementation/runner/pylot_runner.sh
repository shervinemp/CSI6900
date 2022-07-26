#!/bin/bash
# nvidia-docker exec -i -t pylot sudo service ssh start 
# ssh -p 20022 -X erdos@localhost 'cd /home/erdos/workspace/pylot/;export PYTHONPATH="$PYTHONPATH:$PYLOT_HOME/dependencies/lanenet/";python3 pylot.py --flagfile=configs/e2e.conf'

ssh -X -p 20022 erdos@localhost  "cd workspace/pylot && source ./scripts/set_pythonpath.sh && python3 pylot.py --flagfile=configs/e2e.conf --visualize_world"
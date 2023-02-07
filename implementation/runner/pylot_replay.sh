#!/bin/bash
# docker restart pylot >> /dev/null
nvidia-docker exec -i -t pylot sudo rm -f /home/erdos/workspace/pylot/replay.py /home/erdos/workspace/pylot/replay_logs.sh
nvidia-docker exec -i -t pylot sudo rm -rf /home/erdos/workspace/pylot/replay
docker cp "$(readlink -f ../scripts/replay.py)" pylot:/home/erdos/workspace/pylot/
docker cp "$(readlink -f ../scripts/replay_logs.sh)" pylot:/home/erdos/workspace/pylot/
docker cp "$(readlink -f "$1")" pylot:/home/erdos/workspace/pylot/replay
# nvidia-docker exec -i -t pylot sudo service ssh start
# ssh -p 20022 -X erdos@localhost 'cd /home/erdos/workspace/pylot/;export PYTHONPATH="$PYTHONPATH:$PYLOT_HOME/dependencies/lanenet/";./replay_logs.sh "$(readlink -f replay)"'
ssh -p 20022 -X erdos@localhost 'cd /home/erdos/workspace/pylot/;./replay_logs.sh "$(readlink -f replay)"'

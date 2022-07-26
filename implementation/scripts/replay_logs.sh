#!/bin/bash
# $1 Directory where the log files are

#if [ -z "$PYLOT_HOME" ]; then
#    echo "Please set \$PYLOT_HOME before sourcing this script"
#    exit 1
#fi

ADDR=`readlink -f "$1"`

cd ${CARLA_HOME}/PythonAPI/examples

function replay_log {
    FILE_NAME="$1"
    echo "Replaying ${FILE_NAME}"
    HERO_ID=`python3 show_recorder_file_info.py -f "${FILE_NAME}"`
    echo $HERO_ID
    python3 start_replaying.py -f "${FILE_NAME}" -c 2 -x 0.2
    read -p "Press any key when the replay completes... " -n1 -s
}

if [ -f $ADDR ]; then
    replay_log $1
else
    for FILE_NAME in "$ADDR"/*.log; do
        replay_log "$FILE_NAME"
    done
fi

#!/bin/bash
MFILE=/home/erdos/workspace/results/finished.txt
if docker exec -it pylot test -e $MFILE; then
    docker cp "pylot:${MFILE}" ./
fi

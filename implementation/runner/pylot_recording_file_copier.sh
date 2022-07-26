#!/bin/bash
MFILE=/home/erdos/workspace/recording.log
if docker exec -it pylot test -e $MFILE; then
    docker cp "pylot:${MFILE}" ./
fi

#!/bin/bash
PYLOT_ADDR="${PWD%/*}/pylot"
docker network create mynet
nvidia-docker run -itd --name pylot -p 20022:22 -v "$PYLOT_ADDR:/home/erdos/workspace/pylot/pylot" erdosproject/pylot /bin/bash
nvidia-docker cp ~/.ssh/id_rsa.pub pylot:/home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo chown erdos /home/erdos/.ssh/authorized_keys
nvidia-docker exec -i -t pylot sudo service ssh start
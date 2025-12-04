#!/bin/bash
# set -euo pipefail
xhost +local:

IMAGE="dev-container:dev"
CONTAINER_NAME="autolabeling-SAM2-dev"

# this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' is already running. Attaching..."
    docker exec -it "${CONTAINER_NAME}" /bin/bash
else
    echo "Starting new container '${CONTAINER_NAME}'..."
    docker run -itd \
        --rm \
        --gpus all --runtime=nvidia \
        --name "${CONTAINER_NAME}" \
        --privileged \
        --network bridge --ipc=host \
        --env="DISPLAY=${DISPLAY}" \
        --device=/dev/dri \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="/media/q200/kike/docker_ws/docker_utils/:/docker_utils" \
        --volume="/media/datasets/rosbags-bishop-hospital:/data" \
        --volume="/media/q200/models_weights:/models" \
        --volume="/${DIR}:/workspace" \
        -w /workspace \
        "${IMAGE}"
fi

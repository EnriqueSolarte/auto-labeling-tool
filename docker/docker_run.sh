#!/bin/bash
# set -euo pipefail
xhost +local:

IMAGE="auto-labeling:dev"
CONTAINER_NAME="auto-labeling-dev"

# this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

install_completion() {
    docker exec "${CONTAINER_NAME}" bash -lc 'auto-label-tool install-completion'
}

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' is already running. Attaching..."
    install_completion
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
        --volume="/media/q200/models_weights:/models" \
        --volume="/media/v1-data/panda_ai_dataset:/data/panda" \
        --volume="/media/v1-data/hospital-bishop-data:/data/bishop" \
        -w /workspace \
        "${IMAGE}"
fi

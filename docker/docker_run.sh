#!/bin/bash
# set -euo pipefail
# this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${DIR}/env"

if [[ ! -r "${ENV_FILE}" ]]; then
    echo "Error: required development environment file is not readable: ${ENV_FILE}" >&2
    exit 1
fi

source "${ENV_FILE}"
xhost +local:

IMAGE="auto-labeling:dev"
CONTAINER_NAME="auto-labeling-tool"

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
        --volume="${MODELS}:/models" \
        --volume="${DATA}:/data" \
        --env="DATA=${DATA}" \
        -w /workspace \
        "${IMAGE}"
fi

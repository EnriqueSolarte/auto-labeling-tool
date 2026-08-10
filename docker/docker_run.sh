#!/bin/bash
# set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") --data <host-data-directory>" >&2
}

DATA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--data)
            if [[ -z "${2:-}" ]]; then
                echo "Error: $1 requires a directory path." >&2
                usage
                exit 2
            fi
            DATA="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${DATA}" ]]; then
    echo "Error: --data/-d is required." >&2
    usage
    exit 2
fi

export DATA

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
        --volume="${DATA}:/data" \
        --env="DATA=${DATA}" \
        -w /workspace \
        "${IMAGE}"
fi

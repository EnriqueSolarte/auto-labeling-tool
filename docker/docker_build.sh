#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

docker build -f "${SCRIPT_DIR}/dockerfile.auto-labeling" -t auto-labeling:dev "${REPO_ROOT}"

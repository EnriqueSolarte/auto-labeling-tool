# Changelog

## v1.1.0

- Added Docker Compose configuration for the auto-labeling development container.
- Added `.dockerignore` to reduce Docker build context size.
- Updated Docker build and run scripts for the current repository layout.
- Moved `requirements.txt` to the repository root for Docker build compatibility.
- Updated the Docker image to install the package into `/workspace`.
- Documented the Docker Compose workflow in `README.md`.

## v1.0.0

- Added the `auto-label-tool` CLI.
- Added config-driven workflow commands: `create-config`, `create-masks`, `aggregate-masks`, `create-dataset`, `validate`, and `run`.
- Added config version checks against the installed package version.
- Added root-location checks so configs do not run from mismatched `run.root` paths.
- Added Bash completion support through `argcomplete`.

## v0.1.0

- Initial internal alpha release.
- Supported the manual RGB-to-YOLO pipeline through `python -m` app modules.
- Included mask creation with manual prompts and SAM2 propagation.
- Included mask aggregation into `average_mask` outputs.
- Included YOLO dataset generation from aggregated masks.
- Included interactive annotation validation and cleanup.
- Included dataset visualization and dataset stats utilities.
- Included Docker helper scripts for the internal GPU development environment.

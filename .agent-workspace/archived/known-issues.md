Date: 2026-05-14
Project version: v0.1.0
Description: Internal alpha risks and limitations are explicit.

# Known Issues

This project is currently an internal alpha. The following limitations are intentional release caveats for `v0.1.0`, not minor polish items.

## Environment and portability

- Docker startup depends on workstation-specific `.env` host mounts.
- OpenCV GUI windows depend on X11 access from the host.
- The expected SAM2 model location is `/models/sam2`.
- The project assumes Linux, NVIDIA GPU support, and OpenCV GUI availability.
- Docker Compose is provided, but it has only been scoped for the internal alpha workflow.

## Code structure

- The repository is a collection of top-level scripts, not an installable Python package.
- There is no stable public API.
- CLI behavior is split across many scripts instead of one command surface.
- UI, model loading, propagation, filesystem layout, and dataset logic are tightly coupled.

## Reliability

- There are no automated tests yet.
- Some scripts silently skip invalid or mismatched masks.
- Dataset generation does not write a reproducibility manifest.
- Outputs do not record command arguments, git commit, model checksum, threshold settings, or prompt metadata.
- Some defaults still reflect local project history and should be reviewed before use on a new machine.

## Dataset quality

- Mask aggregation assumes object slice ordering is consistent across mask versions.
- Bounding boxes are generated from masks without a real quality filter.
- `compute_affordance_bbx` is currently a placeholder that always accepts boxes.
- The release does not guarantee training quality or annotation correctness.

## Destructive operations

- Dataset pruning and split manipulation can delete or move files.
- Some backup behavior is temporary and not suitable as a safety guarantee.
- Review commands carefully before running destructive modes on important datasets.

## v1.0.0 direction

For `v1.0.0`, this project should become an installable package with a unified CLI, Docker Compose, reproducibility manifests, tests, safer dataset operations, and clearer separation between core logic and interactive UI.

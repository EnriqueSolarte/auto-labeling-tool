Date: 2026-05-14
Project version: v0.1.0
Description: Internal alpha release plan keeps Docker simple and adds release notes.

# Suggested v0.1.0 Release Plan

This release is `v0.1.0`. It is an internal alpha for teammates using Linux, an NVIDIA GPU, Docker, X11, and host-mounted SAM2/data paths.

## Docker scope

- Do not add Docker Compose yet.
- Keep the current `docker/` structure.
- Continue using `docker_build.sh` and `docker_run.sh`.
- Assume Docker builds successfully for this release.

## Changelog policy

Use both files:

- `CHANGELOG.md`: short summary of what this version includes.
- `release-notes/v0.1.0.md`: more detailed notes and planned changes.

Add these as the last release step.

## Docker sanity pass

Original concern: `./docker/docker_build.sh` may depend on the current working directory because it references `dockerfile.autolabeling` without the `docker/` prefix.

Decision for `v0.1.0`: assume Docker is working and already built. Do not spend release time changing Docker unless a blocking issue appears.

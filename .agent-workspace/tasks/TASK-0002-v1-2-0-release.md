# TASK-0002: Release v1.2.0

Status: blocked

Owner: Codex

Parent task: none

Parent milestone: none

## Objective

Prepare, validate, tag, and document the v1.2.0 release.

## Scope

- Update the package version, README, changelog, and release notes.
- Run the full test suite and focused CLI checks.
- Create an annotated local `v1.2.0` tag after release review.

## Out of scope

- Pushing branches, tags, or packages.
- Publishing a hosted release.
- Changing functionality beyond release documentation and metadata.

## Acceptance criteria

- `VERSION`, package metadata, README, changelog, and release notes identify v1.2.0.
- Existing v1.1.0 config files have a documented upgrade path.
- Full tests and release smoke checks pass.
- An annotated `v1.2.0` tag points to the reviewed release commit.

## Repository

| Path | Canonical checkout | Target | Task branch | Baseline |
| --- | --- | --- | --- | --- |
| `/workspace` | `/workspace` | `main` | `release/TASK-0002-v1-2-0` | `5fa3508` |

## Plan

Path: `.agent-workspace/plans/TASK-0002-v1-2-0-release.md`

Current milestone: M2

## Validation

- Installed the declared test extra with `python3 -m pip install -e '.[test]'`.
- `python3 -m pytest`: 36 passed. Pytest could not update its local cache due
  workspace permissions; this did not affect test execution.
- `auto-label-tool --help` and `auto-label-tool visualize-mask --help` passed.
- The historical `tmp/assets__samples/masks` aggregated-mask fixture is absent,
  so its data-specific smoke check was not run.
- Docker is not installed in this environment, so the required Docker build
  check cannot run.

## Next action

Run `docker/docker_build.sh` in a Docker-enabled environment; if it passes,
resume M3 release review and create the approved local tag.

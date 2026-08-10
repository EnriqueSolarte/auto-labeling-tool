# TASK-0002: Release v1.2.0

Status: in-progress

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

Current milestone: M3

## Validation

- Installed the declared test extra with `python3 -m pip install -e '.[test]'`.
- `python3 -m pytest`: 36 passed. Pytest could not update its local cache due
  workspace permissions; this did not affect test execution.
- `auto-label-tool --help` and `auto-label-tool visualize-mask --help` passed.
- The historical `tmp/assets__samples/masks` aggregated-mask fixture is absent,
  so its data-specific smoke check was not run.
- `./docker/docker_build.sh`: passed with Docker daemon access; built and tagged
  `auto-labeling:dev` (image `0816f0ebe14abd114f836f420ca9414f7bd9a3e32edc869ec2c9da4ed8a56cf4`).

## Next action

Review the release commits and create the local annotated `v1.2.0` tag.

## Handoff

Recorded: 2026-08-10

The release work is committed and the working tree is clean on
`release/TASK-0002-v1-2-0`. The release commits ahead of `main` are:

- `84af80f fix: resolve sources before hardlinking`
- `32fdbb7 chore: prepare v1.2.0 release`

Completed evidence:

- `python3 -m pip install -e '.[test]'` installed the package as v1.2.0 and
  installed pytest.
- `python3 -m pytest` passed: 36 tests, with only a non-fatal pytest-cache
  permission warning.
- `auto-label-tool --help` and `auto-label-tool visualize-mask --help` passed.
- `VERSION` and installed package metadata report `1.2.0`.

The initial sandboxed Docker attempt lacked access to `/var/run/docker.sock`.
The approved daemon-access retry passed on 2026-08-10. No `v1.2.0` tag exists,
and no release branch, tag, or package has been pushed or published.

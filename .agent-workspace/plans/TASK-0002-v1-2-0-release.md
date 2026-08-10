# TASK-0002 Permanent Plan: v1.2.0 release

Revision: 1

## Milestones

### M1: Prepare release metadata and notes

Status: completed

Scope: Update the version, changelog, README, and release notes. Document the
config-version migration requirement.

Validation: Review metadata consistency and referenced documentation paths.

Evidence: `VERSION`, README, CHANGELOG.md, and `release-notes/v1.2.0.md` all
identify v1.2.0 and document the config-version upgrade path. Committed in
`32fdbb7 chore: prepare v1.2.0 release`.

### M2: Run release validation

Status: completed

Scope: Install test dependencies, run the full suite, and run focused CLI smoke checks.

Validation: Record exact commands and outcomes in TASK-0002.

Evidence: `python3 -m pytest` passed (36 tests); `auto-label-tool --help` and
`auto-label-tool visualize-mask --help` passed. `./docker/docker_build.sh`
passed with Docker daemon access and built `auto-labeling:dev` as
`0816f0ebe14abd114f836f420ca9414f7bd9a3e32edc869ec2c9da4ed8a56cf4`.

Closeout: The release metadata and behavior were validated so the release can
be reviewed with a reproducible package build. The Docker build resolves the
pinned SAM-2 dependency, installs the project from the repository copy, and
tags the result `auto-labeling:dev`; it confirmed installed package version
`1.2.0`. Evidence comprises `python3 -m pytest` (36 passed), both CLI help
commands, and `./docker/docker_build.sh` (passed). The pytest-cache permission
warning was non-fatal; the historical mask-fixture smoke check remains excluded
because its fixture is absent.

### M3: Review and tag

Status: completed

Scope: Review the release diff and create a local annotated `v1.2.0` tag.

Dependency: M1 and M2 completed.

Closeout: Reviewed the complete `main..HEAD` release series (`84af80f`,
`32fdbb7`, and `5de0ccc`) with `git diff --check`; it passed without whitespace
errors. The reviewed state preserves the v1.2.0 metadata, release notes,
hardlink source-resolution fix, tests, and release-validation evidence. A local
annotated `v1.2.0` tag will point to the dedicated M3 closeout commit. No merge,
push, package publication, or absent-fixture smoke check is included.

## Change log

| Revision | Date | Change | Approval |
| --- | --- | --- | --- |
| 1 | 2026-08-10 | Initial v1.2.0 release plan. | User approved executing the release sequence. |

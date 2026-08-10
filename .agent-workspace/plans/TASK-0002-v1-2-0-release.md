# TASK-0002 Permanent Plan: v1.2.0 release

Revision: 1

## Milestones

### M1: Prepare release metadata and notes

Status: blocked

Scope: Update the version, changelog, README, and release notes. Document the
config-version migration requirement.

Validation: Review metadata consistency and referenced documentation paths.

Evidence: `VERSION`, README, CHANGELOG.md, and `release-notes/v1.2.0.md` all
identify v1.2.0 and document the config-version upgrade path.

### M2: Run release validation

Status: completed

Scope: Install test dependencies, run the full suite, and run focused CLI smoke checks.

Validation: Record exact commands and outcomes in TASK-0002.

Evidence: 36 tests passed; both CLI help checks passed. The only warning was a
non-fatal pytest cache permission warning. Docker is unavailable in this
environment, so `docker/docker_build.sh` remains required before M2 can complete.

### M3: Review and tag

Status: pending

Scope: Review the release diff and create a local annotated `v1.2.0` tag.

Dependency: M1 and M2 completed.

## Change log

| Revision | Date | Change | Approval |
| --- | --- | --- | --- |
| 1 | 2026-08-10 | Initial v1.2.0 release plan. | User approved executing the release sequence. |

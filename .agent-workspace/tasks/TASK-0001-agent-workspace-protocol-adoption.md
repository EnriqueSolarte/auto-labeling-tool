# TASK-0001: Adopt the agent-workspace protocol

Status: integrated

Owner: Codex

Parent task: none

Parent milestone: none

## Objective

Make repository coordination durable and enforceable through protocol-v2 task,
plan, index, and repository rules while preserving the legacy workspace records.

## Scope

- Add the protocol-v2 control-plane records and repository `AGENTS.md`.
- Track `.agent-workspace/` in Git.
- Preserve existing `archived/` and `v1.2.0/` records without rewriting them.

## Out of scope

- Rewriting historical documents into the v2 schema.
- Changing application behavior, CLI behavior, Docker files, or sample assets.
- Merging or pushing this branch.

## Acceptance criteria

- `AGENTS.md`, `active.md`, `PLANS.md`, this task, and its permanent plan agree
  on task identity, owner, repository, branch, status, and current milestone.
- Legacy records remain present and are identified as historical context.
- `.agent-workspace/` is no longer ignored.
- The changes are validated and committed on the recorded task branch.

## Repository

| Path | Canonical checkout | Target | Task branch | Baseline |
| --- | --- | --- | --- | --- |
| `/workspace` | `/workspace` | `main` | `chore/agent-workspace-protocol` | `ec238db` |

## Plan

Path: `.agent-workspace/plans/TASK-0001-agent-workspace-protocol-adoption.md`

Current milestone: M2 (completed)

## Validation

- Verified `AGENTS.md`, task, plan, and derived indexes agree on TASK-0001,
  branch, and milestone state.
- Verified `.agent-workspace/active.md` is not ignored by Git.
- `git diff --cached --check` passed before commit `3f9e4a9`.
- Commit `3f9e4a9 chore: adopt agent workspace protocol` contains the control
  plane and preserved legacy history.
- Merged into `main` as merge commit `5fa3508` after user approval.

## Next action

Archive this integrated task during a future workspace-maintenance task.

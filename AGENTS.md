# Agent Workspace Protocol

This repository uses `.agent-workspace/` as its durable coordination record.

## Required workflow

1. Before non-trivial work, create or update a `TASK-XXXX` record in
   `.agent-workspace/tasks/` with owner, scope, acceptance criteria, repository
   branch, validation, and exactly one next action.
2. Add a permanent plan in `.agent-workspace/plans/` for work with multiple
   slices, dependencies, uncertainty, or approval gates. Keep task records as
   pointers to the current plan and milestone rather than duplicating milestones.
3. Keep `.agent-workspace/active.md` and `.agent-workspace/PLANS.md` synchronized
   as derived indexes. Do not treat them as independent authority.
4. Use the repository's canonical checkout only. Create one task branch per
   repository; do not run overlapping tasks on the same repository in parallel.
5. Preserve legacy records under `.agent-workspace/` as history. Do not rewrite
   their claims to fit the current schema.
6. Record targeted validation and Git evidence in the task before declaring a
   milestone ready for review. Explicit user approval is required before merging
   to `main` or pushing a branch.

## Record lifecycle

`proposed` -> `ready` -> `in-progress` -> `ready-to-merge` -> `integrated` ->
`archived`. A task may be `blocked` only with its blocker and resumption condition
recorded. Plans and completed milestones remain permanent.

## Historical records

The `archived/` and `v1.2.0/` directories predate this protocol. They are
historical context, not active task authority.

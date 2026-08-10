# TASK-0001 Permanent Plan: Agent-workspace protocol adoption

Revision: 1

## Milestones

### M1: Establish the protocol-v2 control plane

Status: completed

Scope: Add tracked protocol records, task authority, a permanent plan, derived
indexes, and repository rules. Preserve legacy records in place.

Acceptance: Task, plan, indexes, and `AGENTS.md` agree; `.agent-workspace/` is
tracked; legacy records are explicitly identified as historical.

Validation: Inspect records, `git check-ignore`, and the staged diff.

Evidence: `git diff --cached --check` passed; commit `3f9e4a9` contains the
protocol-v2 control plane and preserved legacy records.

### M2: Adopt the workflow for future work

Status: pending

Scope: Use a protocol-v2 task and plan for the next non-trivial repository change.

Acceptance: The next task follows the lifecycle and Git evidence requirements.

Dependency: M1 committed and ready for review.

## Change log

| Revision | Date | Change | Approval |
| --- | --- | --- | --- |
| 1 | 2026-08-10 | Initial protocol-adoption plan. M1 completed in `3f9e4a9`. | User approved applying the proposed plan. |

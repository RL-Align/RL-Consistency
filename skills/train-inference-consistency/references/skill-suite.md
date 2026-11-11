# Skill roles, shared authority and handoffs

The suite adds bitwise consistency to an existing model/framework while preserving its required execution and training behavior. It has three independently selectable entrypoints, installed together:

| Entrypoint | Requested outcome | Completion boundary |
|---|---|---|
| [train-inference-consistency](../SKILL.md) | Add or finish the feature end to end | Inspect the complete declared acceptance and deliver measured scope |
| [diagnose-train-inference-consistency](../../diagnose-train-inference-consistency/SKILL.md) | Diagnose drift or verify a change | Reproducible findings and evidence; a failed verification is a valid finding |
| [implement-train-inference-consistency](../../implement-train-inference-consistency/SKILL.md) | Repair or integrate the implementation | Working target-project change with regression evidence and explicit remaining acceptance |

## Installation and authority

Copy all three sibling directories from this revision's `skills/` into the same agent skill directory, or use their paths in a complete repository checkout. Preserve directory names and relative paths. Do not install only a specialist entrypoint, flatten the directories or combine different suite revisions. There is no required runtime package or agent-specific orchestration API.

The `train-inference-consistency/` directory owns the canonical [gates](gates.md), [workflow](workflow.md), [numerical contract](numerical-contract.md), [evidence rules](evidence.md), model guides and templates. Specialists reference these resources directly. Root README and usage pages are user-facing documentation, not hidden runtime dependencies. If a required sibling/reference is absent, report the exact missing path and restore the suite before performing dependent work; unrelated inspection may continue.

The split changes entrypoints and responsibilities, not acceptance semantics. Default C0–C10 acceptance still requires C2/C3/C5/C6/C7/C8/C9/C10 after C0, including both independent training branches, two effective updates and full-model resume. C1/C4 alone are not-required when performance is not requested. Bounded diagnosis or patch verification reports its own scope without relabeling missing gates as passed or optional.

## Iteration and authority to change code

Start from the user's outcome and current evidence. An end-to-end task typically alternates diagnosis, implementation and verification several times. Do not require diagnosis of every operator to finish before testing a first repair. The canonical gate dependencies govern acceptance, not when useful development experiments may begin.

Diagnosis/verification may add task-local tests, capture and reference tooling within the user's scope, preserving product execution semantics and recording observer effects. It does not independently authorize changing arithmetic, cache behavior, routing, objective or deployment. A broader request to implement the feature already authorizes routine transitions into implementation; do not ask again just to change roles. Follow any explicit read-only restriction.

When the user requests only diagnosis or verification, finish with measured findings and the next action even if the result fails. When implementation is authorized, continue a repair and its regressions. Do not recursively invoke entrypoints or bounce between roles without taking a concrete next action. A single agent may follow all procedures; separate agents are optional and require applicable authorization.

## One task workspace and one evidence history

Use one [task brief](../templates/task-brief.md), [run ledger](../templates/run-ledger.md), [comparison matrix](../templates/comparison-matrix.md), append-only [gate attempts](../templates/gate-record.md) and latest [handoff](../templates/handoff.md) per bound scope. A role transition does not create a new acceptance history. A changed model, contract or environment requires the workflow's binding/invalidation rules.

| Transition | Required handoff content |
|---|---|
| Diagnosis to implementation | Exact fixture and source/weight/environment identity; actual failing command/callsites; raw captures and coverage; observed frontier; cause versus hypothesis; requested behavior and repair scope |
| Implementation to verification | Changed files and current source identity; affected paths and stale attempts; original failure plus regressions; exact reproduction commands; actual results and raw artifacts; unsupported cases |
| Verification to implementation or delivery | Inspected identity and evidence; cases actually executed; pass/fail/blocked findings with limitations; next targeted repair or remaining canonical acceptance |

Each recipient verifies bindings and evidence rather than trusting a predecessor's verdict. If a field is unknown, record the gap and next observation instead of fabricating a command or review. Never copy numerical outputs between independent producers, overwrite failed attempts, or treat a handoff narrative as a pass.

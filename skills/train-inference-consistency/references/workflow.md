# Workflow, dependencies and resuming work

Use one task workspace per bound scope in the target project. Pin the model revision, complete weight manifest, tokenizer, input fixture, numerical contract, relevant source/build artifacts, loaded implementations, hardware and process groups. Keep immutable attempt directories beneath it. Markdown is the human index; use suitable machine-readable records and binary tensors in that workspace.

## Dependency graph

```mermaid
flowchart TD
  C0["C0 Inventory"] --> C1["C1 Native baseline"]
  C0 --> C2["C2 Proxy"]
  C2 --> C3["C3 Inference invariance"]
  C3 --> C4["C4 Structure and cost"]
  C4 --> C5["C5 Train–rollout parity"]
  C5 --> C6["C6 Backward integration"]
  C6 --> C7["C7 Independent accuracy"]
  C7 --> C8["C8 Effective updates"]
  C8 --> C9["C9 Full model and resume"]
  C1 --> C10["C10 Delivery"]
  C9 --> C10
```

Develop native tuning separately from consistency implementation, with separate worktrees or equivalent isolated outputs. Share workload/identity requirements, not writable measurements. Use multiple agents only when authorized and useful; the workflow also works sequentially. Early experiments are allowed, but do not label a dependent gate passed while prerequisites remain missing.

## Status and decisions

| Status | Meaning |
|---|---|
| not-started | No execution or implementation attempt yet |
| in-progress | Active work; no accepted result yet |
| pass | Required work and cases executed, evidence inspected, prerequisites valid |
| fail | An executed required check violated its criterion; preserve the failure |
| blocked | A named input, implementation, device, permission or required decision is missing |
| stale | A previously accepted result no longer binds the current implementation/scope |

An unsupported required case is blocked or failed, never passed. A genuinely inapplicable subcase needs an explicit model/scope reason; keep its exclusion visible and do not remove the parent gate. C5/C7/C8/C9 cannot be replaced with a smoke-test exemption. If the user only requests diagnosis or a development subset, report that subset without claiming C0–C10 completion.

Record actual baseline, precision/tolerance, full-model recipe and release decisions. Use existing explicit authorization; do not manufacture reviewer identities or require a new approval for every routine action. Continue independent authorized work while a necessary decision is pending. Baseline and release conclusions must identify who accepted the evidence under the target project's policy.

## Invalidation

Bind each accepted attempt to input/config/source/build hashes, loaded worker identity and the IDs of its prerequisites. A relevant change makes affected attempts stale. Repeated failures remain visible; they do not overwrite a prior success or vice versa. Never repair raw evidence in place.

- Numerical path, casts, reduction order or model mapping changes: revisit C0 and relevant C2–C9 cases.
- Performance/fusion/Graph changes: rerun impacted C3/C5/C7 and downstream full acceptance; measure C4/C10 again.
- Proxy coverage changes: repeat C2 and affected proxy gates.
- Weights, tokenizer, objective, fixture, topology, hardware or provider changes: create a new bound run; reuse only artifacts demonstrably unaffected and explicitly justify reuse.
- Native baseline changes: create a new C1 baseline revision and repeat comparisons that use it. Never silently rewrite a frozen baseline.

## Resume across agents

Before acting, read the task brief, latest run ledger, latest handoff and the raw records for the next prerequisite. Verify relevant working-tree changes and artifact hashes; do not trust `pass` text alone. Record which previous attempts remain valid. Follow the exact next action or explain why new evidence changes it.

A handoff must retain target repo/ref/dirty changes, actual commands, environment/weight identity, coverage gaps, the smallest failing sample, first observed frontier, blockers and next action. If no executable command exists yet, identify the file/function to implement next and mark execution pending. Never invent a command merely to fill a template.

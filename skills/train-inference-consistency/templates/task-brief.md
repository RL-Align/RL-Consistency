# Task brief — fill from the actual target

Status: not-started. Placeholder text is not evidence.

| Item | Observed value / artifact / unresolved question |
|---|---|
| User goal and authorized work | TODO |
| Task boundary and active role | End-to-end, diagnosis-only, implementation, or verification-only; record allowed product changes and requested completion scope |
| Acceptance mode and source | consistency-only by default; record any explicit performance request |
| Target repository paths, refs and local changes | TODO |
| Model ID/revision, config, full weight manifest, tokenizer | TODO |
| Training and rollout frameworks/versions | TODO |
| Hardware, runtime/libraries, actual loaded providers | TODO |
| Dtype, quantization/scales, master/deployment relation | TODO |
| TP/CP/PP/DP/EP groups and physical layouts | TODO |
| Batch/context/padding/packing/cache/Graph/PD ranges | TODO |
| Objective, optimizer, loss normalization and sampling/score semantics | TODO |
| Equality relation(s) requested | TODO |
| Independent accuracy references and predeclared tolerances | TODO |
| Available resources and actual execution constraints | TODO; do not invent a speed target |
| Performance scope / native workload / cost budget | not-required by default; fill only if requested |
| Chosen simple arithmetic path and optional accelerations disabled | TODO; preserve explicitly required behavior |
| Required cases and explicit exclusions with reasons | TODO |
| Existing results, scope and evidence compatibility | TODO |
| Available/missing inputs and next blocker-resolution action | TODO |
| Applicable project decisions and existing authorization | TODO |

## Execution map

Record actual rollout, old-logprob selection/recompute and actor-loss callsites. Identify stored/replayed values, mode switches, synchronization and state ownership. Link inspected files and executed worker evidence.

## Work locations

Name the target implementation worktree(s), immutable attempt output root and, only if enabled, native baseline output root. Keep model code and raw artifacts in the target project. Record exactly which scope changes require a new task/run rather than silently substituting a reference path.

# Canonical C0–C10 stage map

Preserve these stage meanings across models and frameworks. Default to `consistency-only`: C1/C4 keep their performance responsibilities but are `not-required`; all other stages remain required. Choose `consistency-and-performance` only when performance work is requested. An agent narrative or successful process exit alone is insufficient evidence.

| Gate | Scope and prerequisite | Acceptance evidence | Failure routing |
|---|---|---|---|
| [C0](c0-inventory.md) | Start with real target projects | Complete arithmetic/quantization/discrete-choice inventory, actual worker routes, bound identities and acceptance mode | Resolve missing identity or coverage before accepting dependent checks |
| [C1](c1-native-baseline.md) | Optional performance track, after C0 | When enabled: strong native tuning, equal-work benchmark and frozen baseline | Repair baseline/workload; default mode records `not-required` and continues |
| [C2](c2-representative-proxy.md) | C0 | Representative source-preserving layers, real proxy weights and feature/transition coverage | Extend proxy or add explicit fixtures; return to inventory for new operators |
| [C3](c3-inference-invariance.md) | C2 | Required independent repeat/batch/padding/permutation/chunk/PD/cache pairs, raw bytes and coverage | Diagnose divergence, implement numerical fixes and rerun affected cases |
| [C4](c4-structural-performance.md) | Optional performance track, after C3 | When enabled: real profiles and structural-cost repairs | Repeat impacted C3 after changes; default mode records `not-required` and continues |
| [C5](c5-train-rollout-parity.md) | C3; also C4 when enabled | Independently computed old-logprob and grad-enabled actor pairs, actual callsite bindings | Diagnose each branch; repair forward semantics or capture, then repeat impacted gates |
| [C6](c6-backward-integration.md) | C5 | Per-module backward mapping, saved-state/gradient semantics and complete real gradient coverage; cost checks only when enabled | Repair missing or incorrect backward; return to C5 if forward changes |
| [C7](c7-independent-accuracy.md) | C6 | Independent mathematical forward, gradient and optimizer-step comparisons using predeclared tolerances | Retain failing cases and repair numerical/gradient contract; no post-hoc tolerance relaxation |
| [C8](c8-effective-updates.md) | C7 | theta0 compare, effective update, theta1 compare, effective update, theta2 compare; normal exit | Diagnose sync/update/non-finite state; return to affected earlier gates |
| [C9](c9-full-model-resume.md) | C8 and accepted full-model recipe | Full-model dual-path entry, update loop, required context/topology matrix and resumed first forward/update | Refine proxy and repair full-size failure before repeating |
| [C10](c10-delivery.md) | C9; also C1/C4 when enabled | Reproducible implementation, supported-range consistency/accuracy evidence and release decision; performance only when requested | Report missing correctness evidence and reopen affected stages; do not block solely on speed |

Default acceptance order: C0→C2→C3→C5→C6→C7→C8→C9→C10. With performance enabled, C1 branches from C0, C4 is inserted between C3/C5, and C10 joins C1/C9. See [workflow.md](workflow.md) for statuses and invalidation.

A common slow arithmetic implementation is a valid delivered path when integrated and independently executed by rollout, old-logprob and actor. Follow [correctness-first.md](correctness-first.md). Shared computed scores, routing replay, synthetic CPU fixtures and matching final text cannot substitute for the required target-model execution.

Use [gate-record.md](../templates/gate-record.md) and [comparison-matrix.md](../templates/comparison-matrix.md) before running cases. Predeclare precision/tolerance decisions before C7; confirm proxy C5/C7/C8 and the full-model recipe before accepting C9. Do not require native baseline tuning or a training-cost budget in consistency-only mode. Reuse existing user authorization.

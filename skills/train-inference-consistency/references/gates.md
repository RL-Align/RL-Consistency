# Canonical C0–C10 stage map

Preserve these stage meanings across models and frameworks. Read the individual stage instructions before implementation. Each stage requires actual artifacts under the declared scope; an agent's narrative or successful process exit alone is insufficient.

| Gate | Scope and prerequisite | Acceptance evidence | Failure routing |
|---|---|---|---|
| [C0](c0-inventory.md) | Start with real target projects | Complete arithmetic/quantization/discrete-choice inventory, actual worker routes and bound identities | Resolve missing identity or coverage before accepting dependent checks |
| [C1](c1-native-baseline.md) | C0 | Strong native tuning, retained trials, equal-work benchmark configuration and frozen baseline decision | Repair baseline or workload definition; never weaken native to improve the strict ratio |
| [C2](c2-representative-proxy.md) | C0 | Representative source-preserving layers, real proxy weights and feature/transition coverage | Extend proxy or add explicit fixtures; return to inventory for new operators |
| [C3](c3-inference-invariance.md) | C2 | Required independent repeat/batch/padding/permutation/chunk/PD/cache pairs, raw bytes and coverage | Diagnose divergence; implement numerical fixes; rerun affected cases |
| [C4](c4-structural-performance.md) | C3 | Real shapes/call counts, raw profiles, host/workspace/Graph/synchronization assessment | Fix structural defects and repeat impacted C3 |
| [C5](c5-train-rollout-parity.md) | C4 | Independently computed old-logprob and grad-enabled actor pairs, actual callsite bindings | Diagnose each branch; repair forward semantics or capture, then repeat impacted gates |
| [C6](c6-backward-integration.md) | C5 | Per-module backward mapping, saved-state/gradient semantics and training cost | Repair missing or incorrect backward; return to C5 if forward changes |
| [C7](c7-independent-accuracy.md) | C6 | Independent mathematical forward, gradient and optimizer-step comparisons using predeclared tolerances | Retain failing cases and repair numerical/gradient contract; no post-hoc tolerance relaxation |
| [C8](c8-effective-updates.md) | C7 | theta0 compare, effective update, theta1 compare, effective update, theta2 compare; normal exit | Diagnose sync/update/non-finite state; return to affected earlier gates |
| [C9](c9-full-model-resume.md) | C8 and accepted full-model recipe | Full-model dual-path entry, update loop, required context/topology matrix and resumed first forward/update | Full-size failure remains a failure; refine proxy and repair before repeating |
| [C10](c10-delivery.md) | C1 and C9 | Reproduction artifacts, supported-range evidence, equal-work native/strict timing and memory, release decision | Report incomplete delivery or regressions and reopen affected stages |

C1 and C2 are independent branches from C0; C2–C9 are sequential for acceptance. See [workflow.md](workflow.md) for statuses, reviews and invalidation. CPU reference tests, a shared-backend diagnostic, routing replay and a matching final token sequence are useful evidence only for their own measured scope; none substitutes for the required independent production paths.

Use [gate-record.md](../templates/gate-record.md) for attempts and [comparison-matrix.md](../templates/comparison-matrix.md) for predeclared cases. Record precision/tolerance and training budget decisions before interpreting C7/C6 results; confirm proxy C5/C7/C8 and the full-model recipe before accepting C9. Keep decision requirements explicit and reuse existing user authorization where applicable.

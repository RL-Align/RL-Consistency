# C0–C10 gates, workflow v2

Use the gate responsibilities below with the bound model and environment.
Read [the workflow scope](workflow-scope.md) for integration boundaries,
performance measurements and the distinction between diagnostics and acceptance.

| Gate | Required evidence | Exit |
|---|---|---|
| C0 | All-module arithmetic/quantization/discrete-choice inventory, training route, runtime/model identity | Inventory complete; identities match |
| C1 | Strong native baseline, resolved recipe, repeated measurements | Freeze after real human baseline review |
| C2 | Representative proxy with original per-layer dimensions/precision and state interactions | Full feature-to-proxy coverage; list exclusions |
| C3 | Repetition, batch/padding/permutation, chunk, prefill/decode, mixed scheduling and cache-state pairs | Original-dtype raw bytes equal; inference scope only |
| C4 | Real operator shapes/call counts and raw profile hash; host-loop/workspace/Graph checks | Structural defects absent; rerun affected C3 evidence after changes |
| C5 | Separate native old-logprob and grad-enabled actor forward pairs; worker bindings and callsite receipts | Both paths run independently and match rollout at theta0 |
| C6 | Per-module reuse/glue/custom backward inventory, saved forward state, gradient/STE contract, native/strict step timings | Complete backward mapping; within reviewed training budget |
| C7 | Independent forward, gradient and optimizer-step reference comparisons | Reviewed tolerances; reference does not reuse tested backward; SFT quality remains separate |
| C8 | theta0 compare, update, theta1 compare, update, theta2 compare | Two effective master updates, finite/nonzero gradients, normal exit |
| C9 | Reviewed full-model recipe, dual-path theta0 entry, full two-update loop, context/topology and resume evidence | Required full-model matrix passes; proxy results alone do not pass |
| C10 | Delivery consistency evidence and strict/native performance at equal work | Reviewed baseline; report measured scope and overhead |

C1 and C2 branch from C0; C2–C9 are sequential; C10 joins C1/C9. Record precision
review after C0 and before C7. Record `review recipe` after proxy C5/C7/C8 and
before full-model C9 acceptance. Record baseline review before C10 and release
review after all required gates. Reviews record actual decisions, never invented approvals.

Separate master parameters from deployment bytes. Quantization may leave exported
bytes unchanged despite an effective master update. Require a changed complete
master fingerprint, finite nonzero update norm, optimizer state and step evidence;
never use step count alone. Compare each rollout/actor pair at the same version.

Recipe/evidence/run schemas are v2. Do not relabel v1 results, whose C4/C5/C6 meant
different things. Re-export and bind a new recipe, then re-evaluate original raw
artifacts only where the new requirements are met. Missing requirements remain pending.

Failed retries or modified artifacts invalidate descendants. A `rocm-check`
diagnostic report never passes C0–C10, even when every captured token matches.

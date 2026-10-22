# Operator and backward inventory

Bound task/contract revision: TODO. Source/config/loaded worker evidence: TODO.

Create one row per distinct mathematical/operator family and additional rows for numerically different implementations. Cover every model module, embedding/head, scoring/loss, quantization, state and communication boundary. Mark unknowns explicitly.

| Semantic node / layer coverage | Real shape and logical layout | Rollout provider/callsite | Old-logprob provider/callsite | Actor provider/callsite | Arithmetic, precision, casts and reductions | State/discrete choices | Backward and saved state | Reuse/glue/new work | Evidence / blocker |
|---|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Coverage and dispatch

- Expected source modules/families and observed inventory count: TODO.
- Checkpoint/weight converter mapping and tied/shared/expert ownership: TODO.
- Operator dispatch conditions, fallback detection and device support: TODO.
- Process groups, sharding/reconstruction and gradient scaling: TODO.
- Gradient/STE semantics and independent reference for each family: TODO.
- Opaque interiors and missing capture/state coverage: TODO.
- Implemented target files/tests and next missing implementation: TODO.

An inventory of class names without real arithmetic/callsites is incomplete. An intended provider without loaded-worker evidence is not proof of execution.

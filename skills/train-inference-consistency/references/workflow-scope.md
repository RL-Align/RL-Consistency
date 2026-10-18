# Workflow scope and integration boundaries

Apply C0–C10 to one bound model, environment and numerical contract. Use the
user's framework and hardware, preserve the declared comparison scope, and
collect evidence from actual workers. Model preparation and a successful
launcher do not establish bitwise consistency.

Apply these requirements:

- Inventory and proxy first; inference BI/PD is separate from training parity.
- Compare native old-logprob recomputation and the actual grad-enabled actor branch separately.
- Preserve original dtype, masks, token history, weight version and production execution mode.
- Reuse backward under an explicit gradient/STE contract, then use an independent reference.
- Execute two effective master updates with three same-policy comparisons: theta0, theta1, theta2.
- Review the full-model recipe and verify theta0 before multi-step acceptance. Stop escalation on mismatch.
- Keep quality, numerical consistency, normal exit, resume and performance conclusions separate.
- Establish what code, binaries and callsites actually executed.

Keep these implementation boundaries explicit:

| Area | RL-Consistency scope |
|---|---|
| Framework integration | Existing Qwen3/vime/Megatron/vLLM integration; additional framework routes need their own adapters and evidence |
| Numerical providers | Use reviewed backend-specific RL-Kernel providers; inspect actual loaded code and binary artifacts |
| Performance | Replayed equal-work RL-step timing and memory; serving benchmarks need separate implementation and evidence |
| Quality/SFT controls | Report separately; synthetic tests and short logprob runs cannot establish quality |
| Delivery environment | Record actual environment identity; this repository does not build or certify a complete runtime image |

The CPU validators check supplied evidence, not GPU execution authenticity. C4/C6
profile/inventory statements and execution receipts require raw evidence and human
inspection. Report missing producers, runtime providers and validation results
explicitly when describing a model's support status.

For a user-requested ROCm diagnostic on their existing Qwen3 integration, read
`docs/rocm-quickstart.md`: `rocm-plan` is read-only; `rocm-check` explicitly launches
one full-model round at LR=WD=0 through the existing engine. This is a requested
fixed-policy diagnostic, not a shortcut to C9 acceptance. It may execute backward
and optimizer machinery; it proves no effective update. Mark unobserved dual-path
execution, gradients, weights, proxy gates and resume as not verified.

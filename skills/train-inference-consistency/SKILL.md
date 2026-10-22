---
name: train-inference-consistency
description: Develop and verify model training–inference bitwise consistency through C0–C10 using a portable Markdown workflow. Use when an agent must adapt a model or numerical provider, align independent rollout/old-logprob/actor paths, diagnose first divergence, validate batching or prefill/decode behavior, test effective updates and resume, or deliver measured consistency and performance across declared engines and hardware.
---

# Train–Inference Consistency

Implement and verify consistency in the user's target model/framework repositories. Use this Markdown skill as the development procedure. Select providers from the actual environment and numerical requirements; do not require RL-Kernel or a package/CLI supplied by this skill.

Treat “absolute consistency” as original-byte equality within an explicit model, weight version, input domain, precision, execution path and environment. Separate numerical accuracy, performance and quality from equality. Never advertise an unmeasured scope.

## Begin with the actual task

1. Read the target repositories' instructions. Inspect available checkpoints, framework sources, device/runtime details and existing results before asking for missing information. Identify the rollout, old-logprob recompute and grad-enabled actor callsites actually used by the training job.
2. Copy [task-brief.md](templates/task-brief.md), [run-ledger.md](templates/run-ledger.md) and [handoff.md](templates/handoff.md) into a new task workspace in the target project. Resolve their placeholders from inspected evidence. Record blockers instead of inventing values.
3. Read [workflow.md](references/workflow.md), [gates.md](references/gates.md), [numerical-contract.md](references/numerical-contract.md) and [evidence.md](references/evidence.md). Preserve requested engines, caching, batching, topology and gradient behavior. A slower shared execution path can be a diagnostic or explicitly requested implementation; it cannot silently replace the scope being verified.
4. Use the relevant model guide: [Qwen3-8B](models/qwen3-8b.md), [Qwen3.8-27B](models/qwen38-27b.md) or [Qwen3-30B-A3B](models/qwen3-30b-a3b.md). Confirm every architecture fact against the actual checkpoint. For any other model, use [model-extension.md](references/model-extension.md) and [model-profile.md](templates/model-profile.md); registration is not a prerequisite for diagnosis.
5. Continue authorized inspection, implementation, local tests and diagnostics autonomously. When required hardware, weights or a project decision is absent, record precisely what remains blocked and finish independent work. Reuse existing authorization; do not invent approvals or stop for routine implementation choices.

## Execute C0–C10

Read the linked stage before working on it. Record every attempt using [gate-record.md](templates/gate-record.md). Do not treat this table alone as an implementation plan.

| Gate | Read and perform | Required result |
|---|---|---|
| C0 | [Inventory and scope](references/c0-inventory.md) | Complete actual-path inventory and explicit contract |
| C1 | [Native baseline](references/c1-native-baseline.md) | Strong, measured, frozen native configuration |
| C2 | [Representative proxy](references/c2-representative-proxy.md) | Source-preserving reduced model and coverage proof |
| C3 | [Inference invariance](references/c3-inference-invariance.md) | Required independently executed inference pairs match |
| C4 | [Structural performance](references/c4-structural-performance.md) | Actual profile checked; structural defects resolved |
| C5 | [Training–rollout parity](references/c5-train-rollout-parity.md) | Old-logprob and actor forward each independently match rollout |
| C6 | [Backward integration](references/c6-backward-integration.md) | Complete backward mapping and measured training cost |
| C7 | [Independent accuracy](references/c7-independent-accuracy.md) | Independent forward, gradient and update accuracy passes |
| C8 | [Effective updates](references/c8-effective-updates.md) | Two effective updates; theta0/theta1/theta2 comparisons |
| C9 | [Full model and resume](references/c9-full-model-resume.md) | Required full-model matrix and restored execution pass |
| C10 | [Delivery](references/c10-delivery.md) | Reproducible implementation, evidence and fair performance report |

C1 and C2 branch from C0. Develop C2→C3→C4→C5→C6→C7→C8→C9; C10 joins C1/C9. Earlier prototypes can run in parallel, but acceptance must satisfy the recorded prerequisites. Recheck changed artifacts and invalidate affected dependent results.

## Implement, do not only describe

- Create target-repository changes for missing numerical providers, dispatch guards, model mappings, worker capture, independent references and regression cases. Record changed files and actual executable commands in the gate record. The skill does not provide executable stubs or successful-result templates.
- Use [operator-inventory.md](templates/operator-inventory.md) to map every module to real forward/backward implementations, casts, reductions and discrete choices. Apply [numerical-contract.md](references/numerical-contract.md) to select or implement stable arithmetic and reject unsupported fallbacks.
- Use [comparison-matrix.md](templates/comparison-matrix.md) to declare expected cases before running them. Preserve realistic batches, sequence lengths, cache/state transitions, model dimensions and relevant distributed group mappings.
- Use [diagnostics.md](references/diagnostics.md) for repeat checks, semantic DAG frontiers and same-input replay. Preserve failing samples. Equal final outputs cannot rule out intermediate drift; a missing boundary cannot count as equal.
- At C5, execute native old-logprob recomputation and the real grad-enabled actor branch separately. Never copy rollout scores/routes into training or use a detached/inference-only surrogate as actor evidence.
- At C7, keep fixed-order contract checks separate from independently implemented mathematical forward/gradient checks. Do not relax accuracy thresholds after seeing failures.
- At C8/C9, prove effective master updates and compare only within one synchronized weight version. Validate cache invalidation, full-model behavior and the first restored forward/update before claiming completion.

## Record progress and hand off

Use `not-started`, `in-progress`, `pass`, `fail`, `blocked` and `stale` as defined in [workflow.md](references/workflow.md). A report without raw evidence is not a pass. Mark optional cases out of scope only with a recorded scope reason; do not skip mandatory stages to obtain a release.

Update the run ledger and handoff after each significant attempt, including the next command, failing case, relevant commits and artifact paths. Another agent must be able to resume without this conversation. Keep native baseline measurements separate from consistency development.

Use [release-report.md](templates/release-report.md) for delivery. State implemented code, measured equality/accuracy/performance, missing gates and exact supported ranges. Separate “guide available”, “implementation available” and “verified in this environment”. When GPU execution is unavailable, finish useful CPU/reference/tooling work and report GPU validation as blocked; never convert smoke tests into model acceptance.

---
name: train-inference-consistency
description: Coordinate end-to-end bitwise training–inference consistency work in an existing model or training/inference framework. Use when the user wants the feature added and verified, or wants an unfinished consistency task continued. Route bounded diagnosis, verification and implementation to the companion skills without imposing a performance target.
---

# Train–Inference Consistency

Add consistency to the user's actual framework, reusing its working model, training loop, inference engine and weight synchronization where compatible. This entrypoint coordinates the work; the companion skills perform diagnosis/verification and implementation/integration. Runtime code and executable tests belong in the target projects.

## Start from the requested outcome

Read [suite roles and handoffs](references/skill-suite.md). All three sibling skill directories form one portable Markdown suite; preserve their names and relative paths. Reading a companion entrypoint selects a procedure, not a new agent or a separate task.

| User request | Procedure |
|---|---|
| Add or finish consistency in an existing framework | Coordinate the complete task here; iterate between both companions |
| Measure drift, locate a mismatch, or verify a submitted change | Use [diagnose-train-inference-consistency](../diagnose-train-inference-consistency/SKILL.md) for the bounded investigation or verification |
| Implement a consistency fix or integrate compatible arithmetic | Use [implement-train-inference-consistency](../implement-train-inference-consistency/SKILL.md), then inspect its verification evidence |

Do not expand a diagnosis-only request into production changes or a full training campaign. For an end-to-end request, continue through authorized diagnosis, implementation and verification without asking for routine transitions. A missing device or checkpoint blocks only the work that needs it; finish useful independent work and report the limit.

## Establish one task and one contract

1. Inspect the target repositories' instructions, existing run commands, checkpoint, actual worker imports and rollout/old-logprob/grad-enabled actor callsites. Establish what already works and retain the initial failure. This functional check does not activate performance benchmarking.
2. Create or resume [task-brief.md](templates/task-brief.md), [run-ledger.md](templates/run-ledger.md) and [handoff.md](templates/handoff.md) in the target task workspace. Reuse the same records across roles. Verify old evidence before reusing it.
3. Read the canonical [stage map](references/gates.md), [workflow](references/workflow.md), [numerical contract](references/numerical-contract.md) and [evidence rules](references/evidence.md). Bind the scope and predeclare cases with [comparison-matrix.md](templates/comparison-matrix.md). Use [operator-inventory.md](templates/operator-inventory.md) to distinguish reuse, glue and new implementation.
4. Select only the relevant model guide: [Qwen3-8B](models/qwen3-8b.md), [Qwen3.8-27B](models/qwen38-27b.md), [Qwen3-30B-A3B](models/qwen3-30b-a3b.md), or [model extension](references/model-extension.md). Verify architecture facts against the actual checkpoint. For an AMD/vime setup task, consult [ROCm/vime preparation](references/vime-rocm.md); other environments need their own inspected setup.

Absolute consistency means original-byte equality in the declared model, weights, inputs, precision, paths and environment. Independent mathematical accuracy is also required. Shared arithmetic code is allowed; copied scores or replayed model routing do not prove independent execution. Preserve requested engines, cache, batching, topology, objective and gradients.

## Iterate, then accept the complete result

Use the diagnostic procedure to establish a comparable failing case, the implementation procedure to repair the responsible path, and the verification procedure to rerun it and affected cases. Build forward, backward and independent reference checks early; do not wait for a whole model rewrite to discover unusable gradients. Reuse providers that satisfy the contract. [Slow common arithmetic](references/correctness-first.md) is a valid final implementation; it is not a requirement to replace already compatible kernels.

The canonical acceptance order remains **C0 → C2 → C3 → C5 → C6 → C7 → C8 → C9 → C10**. Read the linked gate before accepting it. C1/C4 remain `not-required` unless performance was explicitly requested. Early experiments are allowed; a gate cannot pass without its active prerequisites. Splitting roles does not waive the representative proxy, either independent training forward, real backward, independent accuracy, two effective updates, full-model matrix or restored continuation.

Record each significant attempt in [gate-record.md](templates/gate-record.md), preserve failures and mark affected results stale after changes. A companion's completion means its assigned work is complete, not that the model has passed C0–C10. Use the shared handoff to retain changed files, exact commands, raw evidence, blockers and the next action. No new orchestration framework or multi-agent setup is required.

## Deliver the measured scope

Use [release-report.md](templates/release-report.md). Inspect implementation, raw comparisons, accuracy, updates and resume evidence before claiming complete acceptance. Separate guide availability, implementation availability and measured support. For a bounded task, report its result and the remaining canonical gates without claiming full acceptance.

In the default consistency-only mode, report performance as `not evaluated`. Do not optimize throughput, tune a native baseline or impose a speed budget. Incomplete execution, missing required hardware or failed correctness checks remain explicit blockers or failures, even when a slow implementation is acceptable.

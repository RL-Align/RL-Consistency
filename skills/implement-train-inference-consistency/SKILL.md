---
name: implement-train-inference-consistency
description: Implement and integrate bitwise-consistent arithmetic in an existing training/inference framework. Use when the user authorizes repairing numerical drift, adapting providers or adding consistency while preserving real gradients and weight/state behavior. Reuse compatible code; performance optimization is not required.
---

# Implement and Integrate Train–Inference Consistency

Modify the user's target projects to satisfy the declared numerical contract. Reuse their model loaders, training objective, optimizer, inference engine and weight synchronization where compatible. Prefer focused repairs to replacing a working stack. Code, capture adapters and executable regression tests belong in those target projects, not this Markdown suite.

## Establish the repair boundary

This skill requires the sibling `train-inference-consistency/` directory from the same suite revision. Read [suite roles and handoffs](../train-inference-consistency/references/skill-suite.md), [numerical contract](../train-inference-consistency/references/numerical-contract.md) and [correctness-first implementation](../train-inference-consistency/references/correctness-first.md). Use the shared [stage map](../train-inference-consistency/references/gates.md), [workflow](../train-inference-consistency/references/workflow.md) and [evidence contract](../train-inference-consistency/references/evidence.md), not private acceptance rules. Missing shared references are a packaging blocker.

Read the target repository instructions and task-local brief, ledger and [handoff](../train-inference-consistency/templates/handoff.md). Check the source/weight/environment identities, failing input, actual commands and current captures. Preserve unrelated working-tree changes. An existing handoff is useful but not a prerequisite: if it is absent or stale, use the [diagnostic procedure](../diagnose-train-inference-consistency/SKILL.md) to establish a comparable case and the missing evidence before choosing a repair.

Inspect the actual [operator inventory](../train-inference-consistency/templates/operator-inventory.md), or build it under [C0](../train-inference-consistency/references/c0-inventory.md). Classify compatible reuse, adapter work and new arithmetic/backward work. Use the applicable model guide through [model extension](../train-inference-consistency/references/model-extension.md); adding a registry or adopting RL-Kernel is not a prerequisite. Keep model identity and the user's required engine/cache/batch/topology behavior explicit.

## Make a minimal complete change

1. Identify the earliest confirmed responsible arithmetic or mapping boundary. If the cause is still uncertain, retain the hypothesis and use a targeted experiment instead of rewriting unrelated operators.
2. Implement a compatible common arithmetic path or adapter. Match reduction order, cast/fusion boundaries, state ownership and logical identities where required. Compatible existing providers may remain; slow explicit reductions or sequential evaluation are valid final choices.
3. Bind the implementation to the actual rollout, independent old-logprob and grad-enabled actor callsites required by the task. Shared source code is allowed; shared computed scores or forced model routes are not independent execution.
4. Preserve trainability and inspect saved state, gradient accumulation precision and distributed scaling. Use framework autograd where valid or implement a checked custom backward. Add independent mathematical checks early using [C6](../train-inference-consistency/references/c6-backward-integration.md) and [C7](../train-inference-consistency/references/c7-independent-accuracy.md); matching forward bytes alone is insufficient.
5. Guard unsupported shapes, modes, devices and topologies instead of silently falling back. Verify actual loaded providers and critical configuration, including after worker restart. A flag or registration entry does not prove execution.
6. Preserve the original regression fixture and rerun the changed path, its accuracy/gradient checks and affected integration cases using [diagnosis and verification](../diagnose-train-inference-consistency/SKILL.md). Report failures as failures; do not relax thresholds or alter the objective to make the patch pass.

Implementation and verification are an iterative procedure that one agent can perform. Reading the other entrypoint does not require launching another agent or returning to the coordinator. Continue the authorized task until its requested outcome is achieved or a concrete blocker prevents dependent work.

## Validate integration at the requested scale

For a bounded fix, deliver its regression evidence and identify unexecuted downstream acceptance. For an end-to-end feature request, continue through the complete canonical gates: [C2 proxy](../train-inference-consistency/references/c2-representative-proxy.md), [C3 inference](../train-inference-consistency/references/c3-inference-invariance.md), [C5 training parity](../train-inference-consistency/references/c5-train-rollout-parity.md), backward/accuracy, [C8 updates](../train-inference-consistency/references/c8-effective-updates.md) and [C9 full model/resume](../train-inference-consistency/references/c9-full-model-resume.md).

Verify effective master updates, actual deployment synchronization and invalidation of weight-dependent caches/prepared state. Compare only within the same policy version, preserving the algorithm's frozen-old-policy semantics. Do not turn a small CPU or proxy check into full-model acceptance, or remove a required mode because it is slow. C1/C4 stay not-required unless performance was requested.

## Hand back an executable result

Update the shared [handoff](../train-inference-consistency/templates/handoff.md), [gate record](../train-inference-consistency/templates/gate-record.md) and ledger. Include exact changed files/source identities, rationale, reproduction command and working directory, original failing fixture, implementation choices, tests actually run, raw evidence/digests, affected stale results and remaining blockers. Do not call the implementation complete merely because the patch exists.

Make reproduction safe to repeat: tests that write captures or reports must allocate a fresh attempt directory or require a new output path. The handoff's rerun command must not overwrite artifacts already bound by a manifest. Verify that a second invocation preserves the first attempt's payloads.

For a full delivery, follow [C10](../train-inference-consistency/references/c10-delivery.md) and the [release report](../train-inference-consistency/templates/release-report.md). State the measured supported scope and performance as not evaluated. Another verification pass must be able to inspect the actual implementation and artifacts without relying on this conversation.

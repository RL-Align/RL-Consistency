# C0 — Inventory actual execution and bind the scope

**Entry:** target task and accessible repositories/checkpoint/environment. Read [numerical-contract.md](numerical-contract.md) and [evidence.md](evidence.md).

## Inspect and implement

1. Identify the actual training framework, rollout framework, model classes, loaded kernels/extensions, environment and device groups. Follow the training job from rollout through old-logprob selection/recomputation to the actor loss callsite. Locate flags that bypass recomputation, replay routes or change precision.
2. Inspect checkpoint revision, config, complete weights/tokenizer and framework weight converters. Audit tensor names, dimensions, transposes, tied weights, expert ownership and master-to-deployment casts. Record real hashes and loaded-worker provenance; do not guess missing identities.
3. Fill [operator-inventory.md](../templates/operator-inventory.md) for every module family and boundary, including embedding/head, loss/logprob, communication, normalization, quantization/scales and state updates. Trace actual arithmetic, discrete choices, reductions and backward candidates rather than listing class names alone.
4. Define exactly which relations must match, at which weight version and precision, for which input/context/batch/cache/topology range. Distinguish training/rollout equality from repetition, PD, cross-topology/hardware and optimizer-trajectory equality.
5. Specify the initial numerical contract and acceptance mode: `consistency-only` by default, or `consistency-and-performance` only when requested. Record available execution resources; a speed/training-cost budget is not required in consistency-only mode. Inspect available providers against it; mark reuse, glue, new implementation and missing capture work. Do not assume a library flag or existing model directory establishes compatibility.
6. Implement missing inventory/identity probes and minimal worker-side capture adapters in the target repository. Add a controlled failure check for missing entries or unsupported dispatch. Preserve normal model execution and list any intrusive observer effects.

## Artifacts

Retain the task brief, operator inventory, actual config/weight manifests, logical input fixture, engine/callsite map, numerical contract, provider capabilities/gaps, environment/worker evidence and initial comparison matrix. Record exact commands and files used to obtain them in the gate record.

## Exit

Pass only when every required module/path has an identified implementation or explicitly scoped development task, the accepted comparison scope is unambiguous, and the identities required for downstream execution are available and verified. An inventory may document an unimplemented provider; downstream execution remains blocked until that provider exists. Missing weights or unknown loaded workers prevent binding a completed runtime identity.

If discovery is incomplete, continue useful source inspection and adapter work, record the blocker and leave dependent acceptance pending. Proceed to C2 once its required C0 identities and inventory are valid. Start the independent C1 branch only when performance work is requested. Select a simple implementation using [correctness-first.md](correctness-first.md).

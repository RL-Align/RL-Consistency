# RL-Consistency usage guide

RL-Consistency is a **general-purpose Markdown skill suite** for adding consistency to an existing model or training/inference framework. Give an agent the model, framework sources and hardware environment. Reuse compatible components and follow C0–C10 in the target projects to implement repairs, diagnose divergence, run validation and deliver measured consistency. **The default requires consistency and numerical correctness without a performance target; a slow correct implementation is a valid final delivery.**

This repository supplies methods, development steps, numerical contracts, diagnostics and acceptance templates. Reading the skill requires no Python package from this repository and no RL-Kernel dependency. The target project determines the actual model runtime.

## Install the suite and choose a task

Place all three sibling directories from `skills/` in the same agent skill directory, or read their entrypoints in a complete checkout. Preserve names, referenced Markdown files and relative paths from the same revision:

```text
skills/
  train-inference-consistency/            # Coordinator and canonical shared resources
  diagnose-train-inference-consistency/   # Diagnosis and verification entrypoint
  implement-train-inference-consistency/  # Implementation and integration entrypoint
```

The specialist directories are independently selectable, not independently packaged. Their shared contract, stage references, model guides and templates live under `train-inference-consistency/`. Read [suite roles and handoffs](../skills/train-inference-consistency/references/skill-suite.md) for boundaries and transition requirements.

| Your task | Start here |
|---|---|
| Add the feature or continue an end-to-end task | [Coordinator](../skills/train-inference-consistency/SKILL.md) |
| Locate a mismatch or assess a submitted implementation | [Diagnosis and verification](../skills/diagnose-train-inference-consistency/SKILL.md) |
| Repair a known mismatch or integrate consistent providers | [Implementation and integration](../skills/implement-train-inference-consistency/SKILL.md) |

For diagnosis only:

> Read /absolute/path/RL-Consistency/skills/diagnose-train-inference-consistency/SKILL.md. Diagnose the supplied train/rollout mismatch in FRAMEWORK using CHECKPOINT on HARDWARE. You may add task-local capture and comparison tooling, but do not change product arithmetic or execution behavior. Retain the failing input and raw evidence, distinguish observed frontiers from confirmed causes, and provide the next reproducible action. Do not run a full training campaign or optimize performance.

For an authorized repair:

> Read /absolute/path/RL-Consistency/skills/implement-train-inference-consistency/SKILL.md. Repair the inconsistency described in TASK_WORKSPACE/handoff.md in FRAMEWORK. Verify the handoff against the actual sources, preserve the required cache/batch/topology behavior, and retain real gradients. Run the original failing case and affected correctness regressions; report changes, evidence and any unexecuted downstream gates. Do not optimize performance.

For verification of an existing patch:

> Read /absolute/path/RL-Consistency/skills/diagnose-train-inference-consistency/SKILL.md. Verify the supplied patch and evidence for DECLARED_SCOPE. Inspect actual bindings and rerun the original regression plus affected checks within the allocated resources. Do not repair product code as part of this review. Report failures or missing evidence without converting local results into full-model acceptance.

For AMD validation with vime, follow [ROCm/vime environment preparation and agent task](../skills/train-inference-consistency/references/vime-rocm.md): reuse a suitable current container, or prepare a separate environment when necessary and authorized; inspect prior RL-Kernel integration, then adapt the actual vime/Megatron/vLLM paths. The guide includes copyable Docker commands, environment probes and a complete implementation prompt. Work on Qwen3-8B first, then Qwen3-30B-A3B. Record environment readiness separately from model consistency acceptance.

For the complete feature:

> Use this train-inference-consistency skill to implement training–inference bitwise consistency for MODEL in TRAINING_REPO and ROLLOUT_REPO on HARDWARE. Inspect the actual checkpoint, framework sources and execution paths before following C0–C10. Put code and tests in the target projects. Independently capture rollout, old-logprob recomputation and actor forward. Preserve original failing fixtures and gate evidence. Do not optimize performance. Prefer simple common arithmetic with controllable order, including per-token execution and serial reductions. Mark C1/C4 not-required. Preserve explicitly required model, hardware and execution behavior; leave unexecuted checks pending.

## What C0–C10 requires

| Stage | Required agent work |
|---|---|
| C0 | Inventory actual model/operator/framework/hardware paths; define scope and numerical contracts |
| C1 | Optional native performance baseline; default not-required and does not block consistency |
| C2 | Build a representative reduced-layer model with original weights and original per-layer dimensions |
| C3 | Implement and verify inference repetition, batching, padding, prefill/decode and related invariance |
| C4 | Optional structural performance optimization; default not-required |
| C5 | Independently recompute old logprobs and grad-enabled actor forward; compare each against rollout |
| C6 | Integrate real backward and check saved state, gradient semantics and coverage |
| C7 | Validate forward, gradient and update accuracy against independent mathematics |
| C8 | Execute two real updates and compare same-version paths at theta0/theta1/theta2 |
| C9 | Verify the full model, declared context/topology matrix and first restored forward/next update |
| C10 | Deliver implementation, reproduction, verified bounds and consistency/accuracy evidence |

The default required order is **C0→C2→C3→C5→C6→C7→C8→C9→C10**. C1/C4 retain their original numbers and performance roles; enable them only when the user requests performance work. Missing native benchmarks, profiles or throughput data does not prevent consistency acceptance.

Follow [correctness-first implementation](../skills/train-inference-consistency/references/correctness-first.md). Implement common per-token projections, fixed-order reductions, explicit attention, sequential recurrence, sequential experts and ordered combine in the target projects as needed. Bind these to rollout, independent old-logprob recomputation and the grad-enabled actor. The three paths may share code but must independently recompute. Real gradients, independent accuracy, two effective updates and full-model resume remain mandatory.

The role split does not relax those requirements. C0–C10 is the complete acceptance framework; a bounded diagnosis or repair may finish without completing it, but must identify the remaining gates. Work can alternate between roles before a gate is accepted. Use one task workspace and ledger, preserve failure artifacts and invalidate affected passes after a code change. Keep the current working providers where they satisfy the contract rather than replacing them solely to follow an example.

Each gate defines prerequisites, implementation work, checks, artifacts, acceptance and failure routing. For an unfamiliar model, first locate the earliest observed divergence along the semantic dependency graph, then extend the adaptation.

## Model guides and measured acceptance

| Model | Development guidance | Full-model bitwise acceptance |
|---|---|---|
| Qwen3-8B | Dense attention, GEMM, norm, RoPE, logprob and reductions | One scoped MI300X ROCm/vime/Megatron/vLLM full-model run passed on2026-10-23, including effective updates and exact resume; see [8B measured evidence](../skills/train-inference-consistency/models/qwen3-8b.md#measured-rocm-run--2026-10-23); no extension to other environments/models |
| Qwen3.8-27B | Hybrid text path, convolution/recurrent state and attention interactions | Pending |
| Qwen3-30B-A3B | Independent routing, dispatch, experts and combine | Scoped MI300X full-model comparisons, effective updates and exact restored continuation recorded; see [measured evidence](../skills/train-inference-consistency/models/evidence/rocm-20261023.md#qwen3-30b-a3b). Final delivery bookkeeping is separate from the runtime results |
| Other models | Generic model-extension templates and diagnostics | Determine from implementation and acceptance in the target environment |

Add model-specific Markdown and testing requirements while reusing the common C0–C10 workflow. New operators, quantization and state paths still require code in the target projects. Adding a model name alone does not establish support.

See the [main README](../README.md) for the complete entrypoints. Earlier Python experiments remain on [archive/runtime-v0.3](https://github.com/RL-Align/RL-Consistency/tree/archive/runtime-v0.3); all deliverables on the main branch are Markdown.

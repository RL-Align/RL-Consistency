# RL-Consistency usage guide

RL-Consistency is a **general-purpose Markdown agent skill**. Give an agent the model, training framework, inference framework and hardware environment. The agent follows C0–C10 in the target projects to implement code, diagnose divergence, run validation and deliver training–inference consistency backed by measured evidence. **The default requires consistency and numerical correctness without a performance target; a slow correct implementation is a valid final delivery.**

This repository supplies methods, development steps, numerical contracts, diagnostics and acceptance templates. Reading the skill requires no Python package from this repository and no RL-Kernel dependency. The target project determines the actual model runtime.

## Give the skill to another agent

Place the complete `skills/train-inference-consistency/` directory in the agent's skill directory, or ask it to read [SKILL.md](../skills/train-inference-consistency/SKILL.md) directly. Retain the referenced Markdown files and their relative paths.

For AMD validation with vime, follow [ROCm/vime environment preparation and agent task](../skills/train-inference-consistency/references/vime-rocm.md): create a separate container from an existing image, inspect/remove prior RL-Kernel integration, then modify the actual vime/Megatron/vLLM paths. The guide includes copyable Docker commands, environment probes and a complete implementation prompt. Work on Qwen3-8B first, then Qwen3-30B-A3B. Record environment readiness separately from model consistency acceptance.

Example task:

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

Each gate defines prerequisites, implementation work, checks, artifacts, acceptance and failure routing. For an unfamiliar model, first locate the earliest observed divergence along the semantic dependency graph, then extend the adaptation.

## Model guides and measured acceptance

| Model | Development guidance | Full-model bitwise acceptance |
|---|---|---|
| Qwen3-8B | Dense attention, GEMM, norm, RoPE, logprob and reductions | One scoped MI300X ROCm/vime/Megatron/vLLM full-model run passed on2026-10-23, including effective updates and exact resume; see [8B measured evidence](../skills/train-inference-consistency/models/qwen3-8b.md#measured-rocm-run--2026-10-23); no extension to other environments/models |
| Qwen3.8-27B | Hybrid text path, convolution/recurrent state and attention interactions | Pending |
| Qwen3-30B-A3B | Independent routing, dispatch, experts and combine | Pending |
| Other models | Generic model-extension templates and diagnostics | Determine from implementation and acceptance in the target environment |

Add model-specific Markdown and testing requirements while reusing the common C0–C10 workflow. New operators, quantization and state paths still require code in the target projects. Adding a model name alone does not establish support.

See the [main README](../README.md) for the complete entrypoints. Earlier Python experiments remain on [archive/runtime-v0.3](https://github.com/RL-Align/RL-Consistency/tree/archive/runtime-v0.3); all deliverables on the main branch are Markdown.

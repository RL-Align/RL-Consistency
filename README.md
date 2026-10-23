# RL-Consistency

A **Markdown-only agent skill** for implementing and verifying bitwise training–inference consistency through C0–C10, with **no performance target by default**.

Give this skill to an agent together with a model checkpoint, training/rollout repositories and a hardware target. The agent inventories the actual execution paths, implements the required numerical contracts in those target repositories, diagnoses divergence and collects evidence through full-model updates and resume. The agent may implement slow common arithmetic, sequential reductions or per-token execution; correctness acceptance has no throughput or latency threshold.

The skill has no Python package, bundled kernels, launcher or required RL-Kernel dependency. It can guide work on existing engines or a user-requested custom backend. Reading the skill does not establish consistency; the resulting implementation must pass its declared checks.

[中文说明](docs/README.zh-CN.md) · [Start the skill](skills/train-inference-consistency/SKILL.md) · [C0–C10](skills/train-inference-consistency/references/gates.md)

## Use with another agent

Copy the complete `skills/train-inference-consistency/` directory into your agent's supported skill directory, or give the agent its `SKILL.md` path. Keep its relative Markdown references together. No package installation is needed to read or use the skill; developing and running the target model requires that target's dependencies and hardware.

Example task:

> Use the train-inference-consistency skill in this checkout to develop bitwise consistency for MODEL in TRAINING_REPO and ROLLOUT_REPO on HARDWARE. Inspect the actual checkpoint and loaded implementations, then follow C0–C10. Implement changes and tests in the target repositories, retain independent rollout/old-logprob/actor evidence, and leave unexecuted gates pending. Do not optimize performance. Use the simplest correct implementation, preserve explicitly requested behavior, and mark C1/C4 not-required.

If details are missing, the agent first inspects the accessible target projects, completes the inventory and records the smallest remaining blocker. It does not invent weights, device results or implementation support.

## Workflow

| Stage | Agent outcome |
|---|---|
| C0 | Actual model/operator/engine inventory and comparison contract |
| C1 | Optional native performance baseline; default `not-required` |
| C2 | Representative reduced model with original dimensions and source weights |
| C3 | Inference repeatability, batching and prefill/decode invariance within scope |
| C4 | Optional structural performance work; default `not-required` |
| C5 | Independent old-logprob and grad-enabled actor forward both match rollout |
| C6 | Complete backward integration and actual gradient coverage |
| C7 | Independent mathematical forward, gradient and update accuracy |
| C8 | Two effective updates with theta0/theta1/theta2 comparisons |
| C9 | Full-model matrix and checkpoint-resume validation |
| C10 | Reproducible delivery with scoped consistency and accuracy results |

The default required path is C0→C2→C3→C5→C6→C7→C8→C9→C10. C1/C4 keep their original performance roles but do not block consistency-only delivery. Enable them only when performance work is requested. A slow implementation may pass; copied scores, missing gradients, zero updates or unexecuted full-model checks cannot.

Use the [correctness-first implementation guide](skills/train-inference-consistency/references/correctness-first.md) for concrete construction choices. Each stage specifies prerequisites, code work, measurements, required artifacts, exit conditions and failure routing. The skill also includes a model-independent first-divergence procedure and copyable Markdown records for handoff between agents.

## Model guides and verification status

| Model | Architecture-specific guidance | Full-model bitwise acceptance recorded by this skill project |
|---|---|---|
| [Qwen3-8B](skills/train-inference-consistency/models/qwen3-8b.md) | Dense attention, GEMM/norm/RoPE/logprob and distributed reductions | Pending; reuse prior results only with their exact scope and evidence |
| [Qwen3.8-27B](skills/train-inference-consistency/models/qwen38-27b.md) | Hybrid text path, convolution/recurrent state and attention transitions | Pending |
| [Qwen3-30B-A3B](skills/train-inference-consistency/models/qwen3-30b-a3b.md) | Independent routing, dispatch, experts and ordered combine | Pending |
| Other models | [Generic extension procedure](skills/train-inference-consistency/references/model-extension.md) | Determine from actual implementation and validation |

These are development guides, not installed model implementations or universal guarantees. To add a model, add a Markdown profile describing its verified structure, numerical differences, proxy coverage and test matrix. The receiving agent implements missing code in the target project. A guide alone never marks a model verified.

## Contents and history

All tracked deliverables on `main` are Markdown. The portable skill contains `SKILL.md`, `references/`, `models/` and `templates/`; it needs no files outside that directory. `AGENTS.md` describes maintenance of this documentation repository.

The earlier Python toolkit, shared-prefix runtime, tests and CLI remain available on [archive/runtime-v0.3](https://github.com/RL-Align/RL-Consistency/tree/archive/runtime-v0.3). They are historical experiments, not prerequisites or default execution paths for this skill. Their CPU checks do not establish full-model GPU acceptance.

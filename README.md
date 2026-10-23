# RL-Consistency

A Markdown development and verification guide for agents implementing bitwise training–inference consistency.

The skill's workflow has been applied to develop verified Qwen3-8B and Qwen3-30B-A3B implementations in the environment below. It provides model-specific guidance and measured evidence for further development.

Give your agent the skill, a model checkpoint, your post-training framework and a hardware target. The agent adapts the actual code and verifies independent rollout, old-logprob and grad-enabled actor execution, real updates and checkpoint resume.

[Start the skill](skills/train-inference-consistency/SKILL.md) · [Usage guide](docs/usage.md)

## Verified implementation examples

| Model | Post-training framework | Training–inference consistency | Evidence |
|---|---|---|---|
| [Qwen3-8B](skills/train-inference-consistency/models/qwen3-8b.md) | vime + Megatron; vLLM rollout | Bitwise exact; effective updates and checkpoint resume verified | [Results](skills/train-inference-consistency/models/evidence/rocm-20261023.md#qwen3-8b) |
| [Qwen3-30B-A3B](skills/train-inference-consistency/models/qwen3-30b-a3b.md) | vime + Megatron; vLLM rollout | Bitwise exact; MoE, effective updates and checkpoint resume verified | [Results](skills/train-inference-consistency/models/evidence/rocm-20261023.md#qwen3-30b-a3b) |

Measured on AMD MI300X with ROCm and BF16 models. Here, absolute consistency means identical original bytes within the declared model, weight version, execution paths and environment. The evidence page records the exact implementations, tested ranges and original results.

These examples demonstrate successful implementations within that scope. Runtime code and task-specific patches are retained in the target projects; this repository distributes Markdown guidance. A new agent must implement or adapt the required code and independently validate its results. The examples do not guarantee that every agent can rebuild the implementations from the guide alone.

## Use the skill

Copy the complete `skills/train-inference-consistency/` directory into your agent's skill directory, or give the agent its `SKILL.md` path.

> Use this skill to implement and verify bitwise training–inference consistency for MODEL in TRAINING_REPO and ROLLOUT_REPO on HARDWARE. Preserve independent execution and real gradients, updates and resume. Record evidence and any remaining blockers. Do not optimize performance.

The workflow is designed to guide adaptation to other post-training frameworks, including verl or a custom framework. Each additional framework and environment needs its own implementation and measured validation. See the [ROCm/vime setup guide](skills/train-inference-consistency/references/vime-rocm.md) for the measured stack.

## Contribute

Contribute English Markdown model guides, implementation lessons and measured evidence. Start with the [model extension guide](skills/train-inference-consistency/references/model-extension.md). Keep runtime code and executable tests in the target framework repositories.

This project distributes Markdown guidance with no required RL-Kernel dependency. Earlier runtime experiments remain on [archive/runtime-v0.3](https://github.com/RL-Align/RL-Consistency/tree/archive/runtime-v0.3).

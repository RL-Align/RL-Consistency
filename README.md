# RL-Consistency

A Markdown agent skill for implementing and verifying bitwise training–inference consistency.

Give your agent the skill, a model checkpoint, your post-training framework and a hardware target. It guides the agent through adapting the implementation and verifying independent rollout, old-logprob and grad-enabled actor execution, real updates and checkpoint resume.

[Start the skill](skills/train-inference-consistency/SKILL.md) · [Usage guide](docs/usage.md)

## Measured results

| Model | Post-training framework | Training–inference consistency | Evidence |
|---|---|---|---|
| [Qwen3-8B](skills/train-inference-consistency/models/qwen3-8b.md) | vime + Megatron; vLLM rollout | Bitwise exact; effective updates and checkpoint resume verified | [Results](skills/train-inference-consistency/models/evidence/rocm-20261023.md#qwen3-8b) |
| [Qwen3-30B-A3B](skills/train-inference-consistency/models/qwen3-30b-a3b.md) | vime + Megatron; vLLM rollout | Bitwise exact forward and MoE comparisons; full resume verification pending | [Progress and evidence](skills/train-inference-consistency/models/evidence/rocm-20261023.md#qwen3-30b-a3b) |

Measured on AMD MI300X with ROCm and BF16 models. Results apply to the tested implementations and ranges; the evidence page records exact environments, original results and limitations.

## Use the skill

Copy the complete `skills/train-inference-consistency/` directory into your agent's skill directory, or give the agent its `SKILL.md` path.

> Use this skill to implement and verify bitwise training–inference consistency for MODEL in TRAINING_REPO and ROLLOUT_REPO on HARDWARE. Preserve independent execution and real gradients, updates and resume. Record evidence and any remaining blockers. Do not optimize performance.

The workflow can be adapted to other post-training frameworks, including verl or a custom framework; each implementation needs its own validation. See the [ROCm/vime setup guide](skills/train-inference-consistency/references/vime-rocm.md) for the measured stack.

## Contribute

Contribute English Markdown model guides, implementation lessons and measured evidence. Start with the [model extension guide](skills/train-inference-consistency/references/model-extension.md). Keep runtime code and executable tests in the target framework repositories.

This project distributes Markdown guidance with no required RL-Kernel dependency. Earlier runtime experiments remain on [archive/runtime-v0.3](https://github.com/RL-Align/RL-Consistency/tree/archive/runtime-v0.3).

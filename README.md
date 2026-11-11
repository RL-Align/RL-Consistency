# RL-Consistency

A Markdown skill suite for agents adding bitwise training–inference consistency to existing model and training/inference frameworks. Reuse compatible code and preserve real training behavior; performance optimization is not required.

The skill's workflow has been applied to develop verified Qwen3-8B and Qwen3-30B-A3B implementations in the environment below. It provides model-specific guidance and measured evidence for further development.

Give your agent the skill, a model checkpoint, your post-training framework and a hardware target. The agent adapts the actual code and verifies independent rollout, old-logprob and grad-enabled actor execution, real updates and checkpoint resume.

[Start the skill](skills/train-inference-consistency/SKILL.md) · [Usage guide](docs/usage.md)

## Choose an entrypoint

| Skill | Use it to |
|---|---|
| [train-inference-consistency](skills/train-inference-consistency/SKILL.md) | Coordinate an end-to-end consistency implementation and its acceptance |
| [diagnose-train-inference-consistency](skills/diagnose-train-inference-consistency/SKILL.md) | Measure drift, localize a mismatch, or independently verify a change |
| [implement-train-inference-consistency](skills/implement-train-inference-consistency/SKILL.md) | Repair arithmetic and integrate it into the actual training/inference paths |

The skills share one numerical contract, C0–C10 stage map and evidence history. Diagnosis, implementation and verification can alternate in one agent session; separate agents are not required. A bounded diagnosis can finish with a failure report without changing production code. See [roles and handoffs](skills/train-inference-consistency/references/skill-suite.md).

## Historical verified implementation examples

| Model | Post-training framework | Training–inference consistency | Evidence |
|---|---|---|---|
| [Qwen3-8B](skills/train-inference-consistency/models/qwen3-8b.md) | vime + Megatron; vLLM rollout | Bitwise exact; effective updates and checkpoint resume verified | [Results](skills/train-inference-consistency/models/evidence/rocm-20261023.md#qwen3-8b) |
| [Qwen3-30B-A3B](skills/train-inference-consistency/models/qwen3-30b-a3b.md) | vime + Megatron; vLLM rollout | Bitwise exact; MoE, effective updates and checkpoint resume verified | [Results](skills/train-inference-consistency/models/evidence/rocm-20261023.md#qwen3-30b-a3b) |

Measured on AMD MI300X with ROCm and BF16 models. Here, absolute consistency means identical original bytes within the declared model, weight version, execution paths and environment. The evidence page records the exact implementations, tested ranges and original results.

These examples demonstrate successful implementations within that scope. Runtime code and task-specific patches are retained in the target projects; this repository distributes Markdown guidance. A new agent must implement or adapt the required code and independently validate its results. The examples do not guarantee that every agent can rebuild the implementations from the guide alone.

The [November 11 acceptance record](skills/train-inference-consistency/models/evidence/rocm-20261111.md) tracks the current ROCm validation separately from those historical results. See the [reproduction archive and exact commands](skills/train-inference-consistency/models/evidence/rocm-20261111-reproduction.md), [task status](STATUS.md), and [retained storage investigation](skills/train-inference-consistency/models/evidence/rocm-20261106-storage.md).

## Quick start

1. **Prepare the project and environment.** Download this repository and make the model checkpoint, post-training source, rollout source and allocated GPUs accessible to the agent. For AMD + vime, follow the [ROCm/vime setup guide](skills/train-inference-consistency/references/vime-rocm.md).

2. **Give the agent the complete suite.** Copy all three sibling directories under `skills/` into the same agent skill directory, preserving their names and contents. Alternatively, provide the chosen `SKILL.md` path in a complete checkout. The specialist skills depend on the shared references in `train-inference-consistency/`; do not install them individually or mix revisions.

3. **Provide the task below.** Replace the placeholders with actual paths and hardware details. Let the agent inspect the environment, adapt code in the target projects, run the required checks and retain the implementation and evidence.

```text
Read /absolute/path/RL-Consistency/skills/train-inference-consistency/SKILL.md.
Follow it to implement and verify bitwise training-inference consistency.

Model: <model name>
Checkpoint: <absolute path to original model weights>
Post-training repository: <absolute path>
Rollout repository: <absolute path>
Hardware and allocated GPUs: <device model and allowed GPU IDs>

Inspect the actual sources, weights and execution paths first.
Implement adaptations and executable tests in the target repositories.
Verify independently executed rollout, old-logprob recomputation and
grad-enabled actor forward, real gradients, effective updates and full-model resume.
Retain raw comparison results, exact source/environment identities and a handoff.
Report the verified scope and any remaining blockers. Do not optimize performance.
```

For the measured Qwen workflow, develop and validate Qwen3-8B before Qwen3-30B-A3B. The [usage guide](docs/usage.md) describes the detailed workflow.

Other post-training frameworks, including verl or a custom framework, require their own code adaptation and measured validation.

## Contribute

Contribute English Markdown model guides, implementation lessons and measured evidence. Start with the [model extension guide](skills/train-inference-consistency/references/model-extension.md). Keep runtime code and executable tests in the target framework repositories.

The [suite validation record](docs/skill-suite-validation.md) describes the bounded CPU agent trials for the role split and their limits; these are not model acceptance results.

This project distributes Markdown guidance with no required RL-Kernel dependency. Earlier runtime experiments remain on [archive/runtime-v0.3](https://github.com/RL-Align/RL-Consistency/tree/archive/runtime-v0.3).

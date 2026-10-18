# Inspected upstream sources

Reference repository: https://github.com/RL-Align/RL-Kernel

Workflow requirements and integration boundaries are documented in the
[workflow scope](../skills/train-inference-consistency/references/workflow-scope.md).

Pinned inspection revision: `76a127c206b7ead6fb837b46264ba43f9ef20e82`.

| Source | Reuse or constraint |
|---|---|
| `docs/architecture/repository-layout.md` | Canonical ownership; avoid copying operators or compatibility wrappers |
| `rl_engine/runtime/policy.py` | Strict policy and fallback rules; environment flags alone are not evidence |
| `rl_engine/runtime/plan.py` | P/P, R/R and mixed diagnostic routes |
| `rl_engine/runtime/semantic_registry.py` | Contract/capability checks and implementation provenance |
| `rl_engine/models/qwen3/spec.py` | Official full-model fingerprint cannot be shrunk |
| `rl_engine/validation/models/chain_gate.py` | Existing chain validation; historical C numbers and tolerance judgments must not be conflated with this workflow |
| `rl_engine/entrypoints/repro.py` | Actual launch parser and backend differences |
| `docs/usage/qwen3-vime-consistency.md` | Source setup, profile and independent logprob recomputation |
| `docs/usage/cuda-rocm-consistency-audit.md` | Within-backend scope; historical evidence is not current-revision certification |
| `rl_engine/integrations/orchestrators/vime/experiments/rocm_attention/tis_metrics.py` | Actual `.pt` sidecar schema and rank/call producer, shared by both backend validation paths |
| `rl_engine/integrations/orchestrators/vime/experiments/rocm_attention/validate_artifacts.py` | Non-allgather CP response ownership, mask semantics and train/rollout sidecar comparison |
| `rl_engine/integrations/orchestrators/vime/experiments/dense/validate_run.py` | CUDA sidecar directory, readbacks and aggregate validation; no automatic C0–C10 equivalence |

The RL-Kernel paths above resolve under `https://github.com/RL-Align/RL-Kernel/blob/<revision>/<path>`.
No upstream kernel implementation was copied into this repository. Model config
fixtures and emitted model argument arrays are attributed separately. New
integration work should re-inspect pinned sources and upgrade deliberately.

Qwen3-30B-A3B onboarding also inspects Qwen's HF configuration and vime commit
`d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36`, including its model arguments, native
launcher, routing-replay tests and training-recovery test. The exact source map
and the limits of those references are in the
[MoE guide](../skills/train-inference-consistency/references/qwen3-30b-a3b.md).

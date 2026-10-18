# Inspected upstream sources

Reference repository: https://github.com/RL-Align/RL-Kernel

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

All links can be resolved under `https://github.com/RL-Align/RL-Kernel/blob/<revision>/<path>`. No upstream source was copied into this repository. New integration work should re-inspect the pinned revision before adopting APIs and upgrade pins deliberately.

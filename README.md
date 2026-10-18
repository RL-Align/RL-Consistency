# RL-Consistency

Reusable agent skills and evidence tooling for bitwise train–inference consistency across model integrations, built on [RL-Kernel](https://github.com/RL-Align/RL-Kernel).

**Status: initial development toolkit. No recipe in this repository is GPU-certified.**

**Run on ROCm:** [Qwen3-8B / 8×MI300X quickstart](docs/rocm-quickstart.md).
`rocm-check` now launches a fixed-policy diagnostic and collects raw comparison
results into JSON/HTML. It uses an existing prepared RL-Kernel environment.

**Second model:** [Qwen3.8-27B Hybrid Dense onboarding](docs/qwen38.md).
`prepare-model` validates the real HF config, creates an 8-layer proxy specification
and emits a candidate recipe. The Hybrid strict runtime and GPU verification remain pending.

**Third model:** [Qwen3-30B-A3B MoE onboarding](docs/qwen3-moe.md).
The default 4-layer proxy preserves all 128 experts and top-8 routing. MoE gates
also compare independent expert choices and routing-weight bytes, and check
dispatch/combine receipts. Strict GPU providers and hardware acceptance remain pending.

Workflow v2 aligns C0–C10 responsibilities with [vLLM PR #61551](https://github.com/vllm-project/vllm/pull/61551),
including dual training paths and theta0/theta1/theta2 acceptance. Read the
[explicit adaptations and gaps](skills/train-inference-consistency/references/pr61551.md).
Old v1 evidence is not automatically relabeled or accepted.

This project turns a consistency investigation into a repeatable C0–C10 workflow: freeze a strong native baseline, use a representative reduced model to find numerical problems, compare independently computed rollout/training outputs, check independent references, perform real updates, validate the full model and checkpoint resume, then measure performance.

## What works in this first version

- A CPU-only, standard-library CLI for recipe inspection, immutable run setup, gate evidence validation, review records and HTML reports.
- Raw-byte FP16/BF16/FP32/FP64 comparisons with logical token alignment, signed-zero detection and rejection of empty/non-finite/missing data.
- An explicit worker trace API with fixture coverage checks and loaded module/library fingerprints; optional PyTorch tensor capture.
- A pinned RL-Kernel sidecar importer with TP replica checks and CP token alignment, producing diagnostic reports without certification claims.
- Dependency-aware C0–C10 records; changed evidence or repeated prerequisite gates invalidate downstream results.
- Candidate Qwen3-8B/vime recipes for CUDA SM90 and ROCm gfx942.
- Qwen3.8-27B BF16 text-path preparation: architecture/operator inventory, source-preserving proxy config, CUDA/ROCm candidates and Hybrid-specific gate requirements.
- Qwen3-30B-A3B BF16 MoE preparation, CUDA/ROCm candidates and routing-ledger checks alongside output parity at C3/C5/C8/C9, including checkpoint resume.
- An independent FP64 GDN recurrence/gradient reference for small CPU accuracy fixtures; not a production kernel.
- A Qwen3 command/preflight bridge to a pinned RL-Kernel launcher; it does not reimplement kernels or duplicate framework patches.
- The `train-inference-consistency` skill, including baseline isolation, independent accuracy checks and human review points.

**Not implemented yet:** automatic engine instrumentation/evidence production, proxy checkpoint materialization, a generalized multi-model runtime, a signed verified-recipe registry, or cross-hardware/optimizer-trajectory certification. The Hybrid and MoE adapters implement CPU preparation; they do not install strict runtime providers. GPU jobs, weights and framework environments are supplied by the operator. See [the roadmap](docs/roadmap.md).

## Install and inspect (no GPU needed)

```bash
python -m pip install -e .
rl-consistency list
rl-consistency export-recipe qwen3-8b-vime-rocm --output local.recipe.json
rl-consistency check --recipe local.recipe.json
```

The check prints unbound identities. Populate them from actual source/model/environment artifacts; never guess hashes. Candidate tolerances and coverage minima require review. Recipes are installed as package resources under `src/rl_consistency/recipes/` so the CLI works outside this checkout.

Inspect a launch plan without executing it:

```bash
rl-consistency plan --recipe local.recipe.json \
  --rl-kernel-root /path/to/RL-Kernel \
  --profile /path/to/machine-profile.json --python /path/to/venv/bin/python
```

The plan uses inspected upstream commit `76a127c206b7ead6fb837b46264ba43f9ef20e82`. A source upgrade needs bridge review and new evidence. The default comparison is **within each backend**; CUDA and ROCm recipes are not evidence of equality between those backends.

## Run a verification workflow

After binding real identities and implementing worker-side producers:

```bash
rl-consistency check --recipe local.recipe.json --bound
rl-consistency init --recipe local.recipe.json --run-dir runs/qwen3-001
rl-consistency gate C0 --run-dir runs/qwen3-001 --evidence evidence/c0.json
rl-consistency status --run-dir runs/qwen3-001
rl-consistency report --run-dir runs/qwen3-001 --output runs/qwen3-001/report.html
```

Read [the evidence format](docs/evidence-format.md), [architecture](docs/architecture.md), and [Qwen3 integration](docs/qwen3.md) before writing a producer. A gate never executes arbitrary commands from an evidence file. It snapshots and evaluates supplied artifacts. Missing evidence remains pending; malformed or unsuccessful evidence fails.

Existing RL-Kernel sidecars can now be inspected with `rl-consistency import-rlk`.
See [worker capture and sidecar import](docs/capture.md) for the command, supported
layout and the worker API. The importer retains raw values and source hashes;
it does not convert missing identities into a passing gate.

`rl-consistency run` explicitly launches a GPU job after host preflight. Its log and return code do not automatically pass any gate. `plan`, `check`, `init`, `status`, `compare`, `gate`, `review` and `report` do not launch GPU jobs.

## Skill

The portable skill lives at [`skills/train-inference-consistency/SKILL.md`](skills/train-inference-consistency/SKILL.md). Point your agent's skill loader at that folder, or copy the folder into its configured skill directory. Install the Python toolkit first so the bundled scripts can import it. No personal skill installation or global agent configuration is performed by this repository.

## Tests

```bash
python -m unittest discover -s tests -v
```

CPU tests use synthetic fixtures to verify the tooling. They are not model-consistency results and must never be published as GPU acceptance evidence.

## Trust and release

Evidence files and local reviewer names are not authenticated attestations. Maintainers must inspect provenance and raw artifacts. `release_eligible` means the local workflow and declared reviews are complete; it is not a registry certificate. Editing `status` in a recipe has no certification effect. There is no automatic recipe promotion in this release.

[中文说明](docs/README.zh-CN.md) · [Development plan](docs/roadmap.md) · [Upstream reference map](docs/upstream.md)

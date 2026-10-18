# RL-Consistency

Reusable agent skills and evidence tooling for bitwise train–inference consistency across model integrations, built on [RL-Kernel](https://github.com/RL-Align/RL-Kernel).

**Status: initial development toolkit. No recipe in this repository is GPU-certified.**

This project turns a consistency investigation into a repeatable C0–C10 workflow: freeze a strong native baseline, use a representative reduced model to find numerical problems, compare independently computed rollout/training outputs, check independent references, perform real updates, validate the full model and checkpoint resume, then measure performance.

## What works in this first version

- A CPU-only, standard-library CLI for recipe inspection, immutable run setup, gate evidence validation, review records and HTML reports.
- Raw-byte FP16/BF16/FP32/FP64 comparisons with logical token alignment, signed-zero detection and rejection of empty/non-finite/missing data.
- Dependency-aware C0–C10 records; changed evidence or repeated prerequisite gates invalidate downstream results.
- Candidate Qwen3-8B/vime recipes for CUDA SM90 and ROCm gfx942.
- A Qwen3 command/preflight bridge to a pinned RL-Kernel launcher; it does not reimplement kernels or duplicate framework patches.
- The `train-inference-consistency` skill, including baseline isolation, independent accuracy checks and human review points.

**Not implemented yet:** automatic engine instrumentation/evidence production, automatic reduced-model construction, a generalized multi-model runtime, a signed verified-recipe registry, or cross-hardware/optimizer-trajectory certification. Engine/model protocols are extension interfaces, not working implementations. GPU jobs, weights and framework environments are supplied by the operator. See [the roadmap](docs/roadmap.md).

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

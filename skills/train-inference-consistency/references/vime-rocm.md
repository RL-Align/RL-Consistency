# Apply the skill on AMD ROCm: vime / Megatron / vLLM

Use this procedure for AMD GPUs, vime training and training–inference consistency independent of RL-Kernel. Read [SKILL.md](../SKILL.md) first and retain the default `consistency-only` mode. This document provides environment preparation and an agent task entrypoint; successful installation or ordinary training does not establish bitwise acceptance.

Contents:

- [Responsibilities and paths](#responsibilities-and-paths)
- [Create the container on the host](#create-the-container-on-the-host)
- [Prepare sources and environment records inside the container](#prepare-sources-and-environment-records-inside-the-container)
- [Check whether workspace mounts hide image sources](#check-whether-workspace-mounts-hide-image-sources)
- [Verify actual entrypoints, not only imports](#verify-actual-entrypoints-not-only-imports)
- [Have the agent prepare an editable runtime](#have-the-agent-prepare-an-editable-runtime)
- [Task to give the agent](#task-to-give-the-agent)
- [Execution and delivery](#execution-and-delivery)

## Responsibilities and paths

| Project | Purpose in this task | Where to change code |
|---|---|---|
| RL-Consistency | Markdown C0–C10 procedure, numerical contracts and acceptance rules | Keep this repository Markdown-only; do not install a Python package |
| vime | Organize sampling, training and weight synchronization; integrate independent capture and acceptance | `/workspace/vime` |
| Megatron-LM | Actual training computation for old-logprob recomputation and the grad-enabled actor | A separate source workspace prepared by the agent |
| vLLM | Actual rollout, prefill/decode, KV cache and output capture | A separate source workspace prepared by the agent |
| Common arithmetic | Fixed-order GEMM, attention, norm, logprob, MoE and related operations | Versioned modules in the target projects above; independently executed in all three paths |

First reuse the ROCm software stack in an existing `rl-kernel:rocm-dev` image and inspect/remove RL-Kernel dependencies in a **new container**. The image name does not determine runtime dependencies; uninstalling one package does not prove framework patches and workers are independent. If a reliable independent stack cannot be recovered, use the vime ROCm image or build recipe.

## Create the container on the host

The following is a starting point for one host and one container. Confirm the GPUs allocated to this task and their available VRAM first. `4,5` is only an example: do not claim those devices from this example or assume two devices can hold full MoE training.

```bash
mkdir -p rl-consistency-ct
cd rl-consistency-ct
cat > docker.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

CONSISTENCY_GPUS="${1:?Usage: bash docker.sh GPU_IDS, for example 4,5}"
if [[ ! "$CONSISTENCY_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  printf 'GPU_IDS must be comma-separated integer device indices\n' >&2
  exit 2
fi
CONSISTENCY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONSISTENCY_IMAGE="${CONSISTENCY_IMAGE:-rl-kernel:rocm-dev}"
CONSISTENCY_NAME="${CONSISTENCY_NAME:-rl-consistency-ct}"
mkdir -p "$CONSISTENCY_ROOT/cache/huggingface" "$CONSISTENCY_ROOT/artifacts"
CONSISTENCY_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$CONSISTENCY_IMAGE")"
printf '%s\n' "$CONSISTENCY_IMAGE_ID" > "$CONSISTENCY_ROOT/artifacts/container-image.txt"

docker run -itd \
  --name "$CONSISTENCY_NAME" \
  --privileged \
  --group-add video \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  --network bridge \
  --ipc private \
  --shm-size=64g \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  -v /home/amd/models:/app/model:ro \
  -v "$CONSISTENCY_ROOT:/workspace" \
  -w /workspace \
  -e CONSISTENCY_GPU_IDS="$CONSISTENCY_GPUS" \
  -e HIP_VISIBLE_DEVICES="$CONSISTENCY_GPUS" \
  -e CUDA_VISIBLE_DEVICES="$CONSISTENCY_GPUS" \
  -e VISIBLE_GPUS="$CONSISTENCY_GPUS" \
  -e HF_HOME=/workspace/cache/huggingface \
  -e HF_HUB_CACHE=/workspace/cache/huggingface/hub \
  -e WANDB_MODE=offline \
  --entrypoint /bin/bash \
  "$CONSISTENCY_IMAGE_ID"
BASH

# Replace 4,5 with the GPUs already allocated to this task.
bash docker.sh 4,5
docker exec -it rl-consistency-ct bash
```

Use separate networking, IPC and the default private PID namespace so that Ray ports, shared memory and process management remain independent of older tasks. `64g` is a shared-memory limit adjustable to host RAM, not GPU VRAM. Cross-container or multi-node tasks require separate network configuration; this recipe does not claim that scope. Inspect inherited ROCm environment requirements before changing them.

Mount original models read-only at `/app/model`. Write new downloads, converted weights, sources and results under `/workspace`, retained in the host workspace. GPU visibility variables select devices in software; they are not hardware isolation. Verify actual device mappings in Ray, training and inference workers. Do not add `--pid=host` or clean up old containers or host processes. Reenter an existing container with `docker exec`; this script does not automatically delete a container with the same name.

Check each real worker's visible-device list and UUID/PCI mapping. Some ROCm HIP-IPC compatibility code places the engine's own device first and appends other devices. Appended IDs must come from the inherited allocated list, not `range(num_gpus_per_node)` interpreted as physical GPU IDs. With nonzero or noncontiguous allocations, a node's device count does not authorize exposing devices starting at zero. Reject a base ID outside the allocation or insufficient allocated devices, then verify fresh workers. Logical `cuda:0` alone does not identify the physical card.

## Prepare sources and environment records inside the container

Run this only inside the newly created container. A cloned vime checkout is a source candidate to inspect; do not immediately upgrade dependencies in an unknown older image.

```bash
cd /workspace
git clone https://github.com/RL-Align/RL-Consistency.git
git clone https://github.com/vllm-project/vime.git
git -C /workspace/vime switch -c consistency/rocm

export CONSISTENCY_BOOTSTRAP="$(mktemp -d /workspace/artifacts/bootstrap.XXXXXX)"
git -C /workspace/RL-Consistency rev-parse HEAD > "$CONSISTENCY_BOOTSTRAP/skill-commit.txt"
git -C /workspace/vime rev-parse HEAD > "$CONSISTENCY_BOOTSTRAP/vime-candidate-commit.txt"
python3 -m pip list --format=json > "$CONSISTENCY_BOOTSTRAP/packages.json"
set -o pipefail
python3 - <<'PY' | tee "$CONSISTENCY_BOOTSTRAP/probe.txt"
import importlib
import os
import torch

print("torch:", torch.__version__, "HIP:", torch.version.hip)
for key in ("CONSISTENCY_GPU_IDS", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES", "VISIBLE_GPUS"):
    print(key, "=", os.environ.get(key))
assert torch.version.hip, "The loaded PyTorch is not a ROCm build"
assert torch.cuda.is_available(), "No usable GPU in this container"
requested = os.environ["CONSISTENCY_GPU_IDS"].split(",")
assert len(set(requested)) == len(requested), "Duplicate GPU indices"
assert torch.cuda.device_count() == len(requested), "Inspect visibility/remapping"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("logical device", i, p.name, getattr(p, "gcnArchName", "unknown"), p.total_memory)
    x = torch.ones((16, 16), device=f"cuda:{i}")
    assert torch.isfinite(x @ x).all().item()
for name in ("vime", "vllm", "megatron.core", "ray", "transformers"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "unknown"), getattr(module, "__file__", None))
    except Exception as exc:
        print(name, "IMPORT_FAILED", type(exc).__name__, str(exc))
PY
```

This is an environment probe, not C3/C5 acceptance. If a framework reports `IMPORT_FAILED`, record and repair the environment before starting full training. PyTorch on ROCm still uses the `torch.cuda` API. The observed `gcnArchName` determines the architecture needed for compilation; do not replace an actual `gfx942` target merely because another script targets a different AMD device.

The agent can now enter the container, use `/workspace` as its working directory and read the task below. An agent installed on the host can use `docker exec rl-consistency-ct ...` for container commands. Final acceptance must run in the recorded container on the recorded devices.

## Check whether workspace mounts hide image sources

A failing `import vllm` inside the container does not necessarily mean the image lacks vLLM. The `-v "$CONSISTENCY_ROOT:/workspace"` mount above hides the image's entire original `/workspace`. Editable installs or `easy-install.pth` files may still point to the hidden source tree. Inspect `.pth`, `.egg-link` and editable-finder targets before deciding to install or compile. `pip list` or `find_spec` alone cannot distinguish a missing package from a broken binding.

Record the target container's exact image ID on the host. Inspect original paths with a temporary container that has no GPU access, workspace mount or network access. For example, this inspection does not run a model:

```bash
CONSISTENCY_IMAGE_ID="$(docker inspect --format '{{.Image}}' rl-consistency-ct)"
docker run --rm --network none --entrypoint /bin/bash "$CONSISTENCY_IMAGE_ID" \
  -c 'ls -la /workspace'
```

If the required sources are confirmed to exist in the image, create a **stopped helper container unique to this task** and use `docker cp` to recover the verified source directory and compiled extensions into a separate workspace. The destination must not already exist, preventing overwrites of prior changes. Preserve Git commits, original patches and extensions. Check the APIs required by the current vime revision and verify loading inside the target runtime container.

Copying sources does not establish API/ABI compatibility. Recovered files may have a different owner from the runtime user. Resolve Git ownership checks only for verified task-owned paths; do not disable checks globally. Afterwards, remove only the helper container created by this task.

When recovering Megatron sources from a package installation, retain original dist-info and per-file content identities. If an upstream SHA is unavailable, record it as unknown; a version number is not a commit. Confirm that both `megatron.core` and `megatron.training` load from the independent directory. Appending a namespace path alone may still prefer the old site-packages implementation.

Run probes in fresh processes or script files, not only with `python -c` at the repository root: the current directory can make an uninstalled vime checkout appear importable. After binding sources through the editable installation below, rerun the independent probe and verify real workers again. For missing upper-layer dependencies such as `vllm_router`, first confirm the package name in the selected vime requirements, then install an explicit version without replacing existing ROCm packages. Preserve original failures and installation logs.

## Verify actual entrypoints, not only imports

An importable Megatron wheel may still lack submodules loaded only during model construction, such as `megatron.post_training`. Test the selected vime conversion/model-construction entrypoint. On failure, inspect the complete source revision pinned by that vime revision's ROCm recipe. Preserve the original wheel snapshot and failure evidence before binding complete sources. Review AMD patches individually: device repairs and semantic changes allowing missing weights or partial loading are separate decisions. Never silently weaken weight completeness to make startup succeed. With unfused attention selected, check inherited Transformer Engine switches so the CLI does not conflict with settings such as `NVTE_FUSED_ATTN`.

vLLM frontend CLI locations, health endpoints and sampler/runner types change across revisions. Probe actual source capabilities and cover both argument discovery and real server startup. If an older generation health endpoint explicitly returns 404, use a real one-token generation probe on the verified interface. A liveness response does not prove generation readiness, and 503 or generation failure must not be swallowed as success. Confirm capture target types and execution counts in real workers. Switching legacy/v2 runners changes scope; evidence for one runner does not establish the other. An unused optional transport should not block the default path through an unconditional import. Repair conditional imports and test the missing-package case instead of installing unverified dependencies throughout the image.

Verify that settings actually reach workers, not only the driver CLI. vime's server environment construction may override `VLLM_USE_V2_MODEL_RUNNER`; a fixed argument dictionary may ignore an explicit `--vllm-logprobs-mode`. Record the actual worker runner type and sampler `logprobs_mode`, then check them against the declaration. Preserve project defaults while enabling explicit choices, and test both behaviors. Temperature1 alone does not authorize treating processed and raw scores as the same acceptance semantics. Prove equivalent processing or rerun the actual raw branch. Old evidence for another runner or scoring semantics cannot inherit a pass in the new scope.

## Have the agent prepare an editable runtime

1. **Freeze the actual environment.** Inspect original vime/Megatron/vLLM load paths, source commits, uncommitted edits, compiled extensions and necessary AMD patches. Record ROCm/PyTorch/Transformer Engine/RCCL/Ray versions, device mappings and relevant environment variables. Log task-related settings without credentials. Check model snapshots, configuration, tokenizer and weight shards, resolve cache symlinks and record content identities.
2. **Align sources and dependencies.** Prefer the image's working stack and corresponding sources as the starting point. Use a matching vime revision and retain useful AMD patches and provenance; do not discard uncommitted image patches. Prepare independent Megatron/vLLM workspaces when modifications are needed, then rebuild or bind them using that revision's ROCm development procedure. Pairing latest vime with unknown older vLLM APIs does not establish readiness.
3. **Remove RL-Kernel integration.** Uninstall `rl-kernel` in the new container. Inspect actual sources, launch entrypoints, environment injection, `.pth`/`sitecustomize`, import hooks, shared libraries and framework monkey patches. Remove or replace dispatch that depends on it. Preserve necessary ROCm/Torch Memory Saver repairs; do not indiscriminately clear `LD_PRELOAD` or `PYTHONPATH`. Verify loaded modules, numerical providers and source paths in newly started real workers. Old workers cannot serve as post-uninstallation evidence.
4. **Bind independent workspaces.** Run the following only after checking versions and build dependencies. `--no-deps` prevents automatic replacement of ROCm dependencies; it does not repair incompatible versions. Recheck real worker imports after binding. Image `.pth` files, Ray `runtime_env` or hard-coded `/root/Megatron-LM` paths may still select older sources.

   ```bash
   python3 -m pip uninstall -y rl-kernel
   python3 -m pip install --no-deps --no-build-isolation -e /workspace/vime
   ```

5. **Prepare inputs and launch entrypoints.** Prefer existing original Qwen3-8B/Qwen3-30B-A3B weights under `/app/model`; do not assume they are ordinary directories or require downloading again. Use the selected vime revision's converter to create Megatron weights in a separate directory under `/workspace/artifacts`. Verify logical tensor mapping and numerical identity. If data is missing, follow that revision's training examples for the actual objective. Record tokenizer, chat template, response masks and loss settings. Create task launch scripts with allocated GPUs, unique output directories and task-owned processes. Do not execute generic upstream `pkill` commands or erase old checkpoints.
6. **Check the minimum runnable chain.** Confirm model loading, one rollout, training forward/backward and one weight synchronization. Ordinary success only resolves environment blockers; it is not a native performance baseline and does not replace bitwise or accuracy gates. Then implement and accept C0–C10.

If the image lacks a compatible stack or existing integration cannot be removed, read the selected vime revision's [AMD guide](https://docs.vllm.ai/projects/vime/en/latest/platform_support/amd_tutorial.html), [ROCm Dockerfile](https://github.com/vllm-project/vime/blob/main/docker/Dockerfile.rocm) or [ROCm 7.14 Dockerfile](https://github.com/vllm-project/vime/blob/main/docker/Dockerfile.rocm7.14). Select a recipe compatible with the hardware and host driver. Record its commit, actual base image ID, dependency revisions and patches when preparing a new image; a floating `latest` is not a reproducible identity. Successful building/importing still does not establish consistency.

## Task to give the agent

Give the following task to an agent with access to this container and `/workspace`:

```text
Read /workspace/RL-Consistency/skills/train-inference-consistency/SKILL.md,
references/vime-rocm.md, references/correctness-first.md and the model guides.

Goal: in this AMD ROCm container, use vime as the upper framework, Megatron
as the training path and vLLM as the rollout path. Implement and verify
training–inference bitwise consistency for Qwen3-8B first, then Qwen3-30B-A3B.
Do not optimize performance or depend on RL-Kernel.

Inspect the actual environment, models and allocated GPUs first. Prepare
matching sources/dependencies and separate editable workspaces. Original
models are under /app/model; write caches, conversions and results under
/workspace. Use /workspace/vime and modify independent Megatron/vLLM source
workspaces as needed. Put implementation/tests in target projects and keep
RL-Consistency Markdown-only.

Execute authorized environment repairs, code changes, tests, diagnosis and
reruns; do not stop at a plan. Record exact missing hardware, weights or
resource blockers and continue work that does not depend on them.
C1/C4 are not-required. Per-token computation, fixed-order reductions,
explicit attention, sequential experts and common operator code are allowed.
Retain three real entrypoints and independently recompute in all three paths.

Before running, declare model/weight versions, precision, topology, external
batch/sequence ranges, prefill/decode, KV cache and the objective. Keep real
vime training and the grad-enabled actor; standalone HF/CPU comparisons do
not substitute. Test actual cache behavior wherever caching is required.
A representative reduced-layer model is allowed during development;
final acceptance requires the full original model.

Follow C0→C2→C3→C5→C6→C7→C8→C9→C10. Independently execute rollout,
old-logprob recomputation and actor forward; retain original bytes and
intermediate boundaries. Replay the first differing actual provider with
the same input, repair it and rerun affected checks. Validate independent
mathematical accuracy, real gradients and two effective updates. Compare
same-version paths at theta0/theta1/theta2. Complete full-model execution and
the first restored forward plus next actual update after checkpoint loading.
For MoE, independently compare router logits, expert IDs/weights, dispatch
and combine. Never replay rollout routing decisions to fake independence.

Retain executable installation, launch, test and reproduction commands with
version/evidence identities. Update ledger/handoff after each gate. Deliver
separate acceptance tables for both models. Never mark unexecuted, incomplete
or failing cases as pass. Matching generated text, near-zero KL, allclose,
a shared seed or one successful training run does not replace raw-byte checks.
```

For the inspected vime revision, pay particular attention to these locations. Resolve actual call relationships again for other revisions:

| Location | Behavior that must be verified |
|---|---|
| `vime/backends/megatron_utils/actor.py` | A genuinely separate old-logprob forward; inspect `can_reuse_log_probs_in_loss` and similar reuse branches |
| `vime/backends/megatron_utils/loss.py` | Whether rollout scores or the current actor's `detach()` are used as old scores; acceptance capture must execute independently |
| `vime/rollout/vllm_rollout.py` | Actual rollout-worker input, weight version and unmodified numerical output capture |
| `vime/backends/megatron_utils/update_weight/` | Training-to-inference tensor mapping, synchronization completion, versions and cache invalidation |
| Actual Megatron/vLLM model and operator sources | Common numerical-provider binding, unsupported-fallback rejection and gradient completeness |

Setting `use_rollout_logprobs=False` alone may not trigger an independent old forward: some paths reuse current actor scores. Explicitly execute, bind and capture the independent old branch for acceptance cases while preserving frozen old-policy semantics in the real RL objective. Use newly synchronized cases at theta1/theta2 for same-version comparisons; do not rewrite old trajectories' old-policy scores with a newer policy. Inspect configuration and interfaces in the current revision instead of inventing a one-switch consistency option.

## Execution and delivery

Proceed through environment readiness → representative Qwen3-8B layers → full Qwen3-8B → representative Qwen3-30B-A3B layers → full Qwen3-30B-A3B. Reuse verified common implementations only when their numerical contracts remain compatible. Independently accept MoE routing, experts, combine, gradients and the full model. If full-model memory is insufficient, continue proxy development and keep the full model blocked; reducing experts cannot establish acceptance of the original model.

Require delivery in the target workspace: source commits/patches, reproducible installation steps, actual launch scripts, input/weight identities, gate records, independent raw captures, first-divergence diagnosis and full-model/resume reports. Convert temporary site-packages edits into reproducible source changes. Record compiled extensions and actual load paths so repairs do not disappear on restart.

| Distinct result | What it establishes |
|---|---|
| Environment probe and minimum native chain succeed | The software stack runs; bitwise consistency remains unproven |
| Proxy passes | Corresponding operator/layer paths are verified; the full model remains pending |
| All required gates pass for one model | That model completes consistency-only acceptance for the recorded weights, devices, precision, topology and inputs |
| Both models pass | Two independent accepted scopes; no extension to every model, AMD device or configuration |

Use [release-report.md](../templates/release-report.md) separately for both models. C1/C4 are `not-required` and performance is `not evaluated`. This guide supplies an implementation route; the required operator work, completion time and feasibility on current resources must be determined from actual execution.

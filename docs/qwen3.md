# Qwen3 integration

For the runnable ROCm fixed-policy diagnostic, start with
[rocm-quickstart.md](rocm-quickstart.md). The generic bound-recipe launcher below
is a separate tool for already reviewed experiments, not full acceptance automation.

The bridge calls `rl_engine/entrypoints/repro.py run` at the inspected RL-Kernel commit. It passes an explicit profile, runtime Python, backend, mode, training/rollout TP/CP, sampling, seeds, step count, LR, weight decay, KL, response cap and microbatch token budget. It does not install packages, download checkpoints, apply patches, start a generic model adapter or mint verification evidence.

Prepare the backend-specific RL-Kernel environment and companion patches using upstream documentation. Do not mix the CUDA and ROCm patch sets. Keep all companion source checkouts clean and pinned.

Host preflight checks the checkout commit/cleanliness, profile hash, explicit framework paths/commits, model config hash, imported RL-Kernel location, package lock, device backend/count/architecture and explicit Python identity. Record weights/tokenizer identity and actual extension/operator provenance inside workers; host preflight alone cannot observe distributed execution or weight synchronization.

```bash
rl-consistency preflight --recipe local.recipe.json \
  --rl-kernel-root /path/to/RL-Kernel \
  --profile /path/to/machine-profile.json --python /path/to/venv/bin/python

# Explicitly launch a GPU job; a successful job does not automatically pass a gate.
rl-consistency run --recipe local.recipe.json \
  --rl-kernel-root /path/to/RL-Kernel \
  --profile /path/to/machine-profile.json --python /path/to/venv/bin/python \
  --steps 2 --output runs/launcher-001
```

Review the native and strict plans before launching. `--mode native` is the baseline path. Only one job should occupy the allocated device set at a time. The upstream launcher manages its backend-specific validation and Ray behavior. Logs go to `launcher.log`; upstream run paths are recorded in that log. Do not terminate unrelated services to free resources.

Supported bridge scope: eight GPUs, training TP×CP=8, training/rollout PP=EP=explicit DP=1; rollout TP×CP divides eight. ROCm additionally requires rollout CP1, positive temperature and top-k disabled. The bridge uses `run` on both platforms, not CUDA-only `verify`. Two launcher steps alone do not prove two nonzero updates; C8 requires separate actual optimizer evidence.

The parent Python process may pass host preflight while engine subprocesses import different modules. Capture worker provenance in C0/C5 and confirm all relevant ranks. Successful launch planning is not a source/build/runtime equivalence proof.

## Existing run artifacts

`rl-consistency import-rlk` can inspect the pinned CUDA `mismatch-sidecars` or
ROCm `mismatch_sidecars` directory after a run completes. It reconstructs
response-token positions, checks TP replicas and CP coverage, and retains raw
bytes and source hashes. It produces diagnostics only: these sidecars omit
the full identities needed for formal gate evidence. For the command and
worker-side capture API, see [capture.md](capture.md).

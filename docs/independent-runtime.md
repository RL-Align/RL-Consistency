# Independent shared-prefix runtime

`runtime-check` uses PyTorch and pinned Transformers 5.20.1, with no RL-Kernel runtime, checkout, profile or patch requirement. It loads local safetensors and executes actual model code. CPU integration tests use PyTorch 2.6.0; other Torch/backend combinations must be measured separately. The dependency extra deliberately leaves the platform Torch installation to the operator.

## Numerical contract

For a token at logical position t, every phase receives exactly the same unpadded prefix, position IDs, all-one attention mask, weight version and vocabulary. Physical batch size is always one; prefix length is determined by that logical token, not the surrounding request list. Old-logprob recomputation reverses request order to exercise this property.

The path uses eager attention and eager experts, no cache, no packing, no autocast, no gradient checkpointing, no hub kernels and one CPU/GPU device. Supported parameter types are FP32/BF16, including provider-required FP32 Hybrid parameters. All phases are eval-mode, autograd-enabled forwards. Eval disables dropout; it does not disable gradients. Generation detaches its own observations; actor creates and differentiates a fresh graph. Calling from `inference_mode` fails.

The scoring rule is FP32 `logit[token] - logsumexp(all vocabulary logits)`. Generation uses greedy argmax and a fixed token count, deliberately continuing after EOS for calibration. This is the raw model policy logprob, not the probability under a greedy sampling distribution. There is no temperature/top-p or stochastic generation implementation in this version.

Every checked token compares:

1. Ordered decoder-layer output hashes (original dtype/shape/bytes).
2. MoE router logits, selected expert IDs and post-cast weights, from actual router outputs in each phase.
3. Full vocabulary logits in their original dtype.
4. The selected-token FP32 logprob, without numerical tolerances.

A decoder hash mismatch identifies the first **observed layer boundary**, not the first internal operator or a proven cause. Use the semantic [diagnostic matrix](diagnostics.md) for finer capture/replay. Router capture does not provide the legacy distributed dispatch/combine ledger, nor impose a new top-k tie policy on the provider.

`torch.use_deterministic_algorithms(True, warn_only=False)` runs forward, scoring, backward and optional optimizer steps. TF32 and cuDNN benchmarking are disabled during execution, and prior global settings are restored afterward. CLI sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization; API callers on NVIDIA must set it before starting Python. ROCm does not use this CUDA-only setting. Run this reference executor from one thread; its temporary global Torch settings are not thread-local.

Model parameters and buffers are hashed before/after phases. Observations never cross weight versions. A non-finite tensor, unexpected checkpoint key, missing checkpoint weight, changed state, unsupported provider or differing byte fails. Hybrid's provider decorators can activate installed FLA/causal-conv implementations even with hub kernels disabled; the reference environment therefore excludes those packages. All families exclude `kernels`. Use a dedicated environment instead of uninstalling packages from a shared training environment.

These controls and observed comparisons do not constitute a proof for arbitrary inputs or environments. PyTorch deterministic checks only cover operations it knows how to check. Records include checkpoint/source/native-extension hashes and package/device identity, but are not authenticated complete GPU binary attestations.

## Commands and artifacts

From the repository root, after installing the platform PyTorch:

```bash
python -m pip install -e '.[runtime]'
rl-consistency runtime-models
rl-consistency runtime-check \
  --model Qwen/Qwen3-30B-A3B --checkpoint /models/Qwen3-30B-A3B \
  --fixture examples/runtime-tokens.json --device cuda:0 --dtype bfloat16 \
  --max-new-tokens 2 --updates 2 --output runs/moe-independent-001
```

The token fixture schema is `rl-consistency.token-fixture.v1` with a nonempty `requests` list of unique `id` and nonempty `input_ids`. IDs must fit the actual vocabulary. `--prompt TEXT` instead loads the local tokenizer without remote code and records tokenized input. No model downloads or remote code execution occur.

Each output directory is new. `theta-N-rollout.json` is published before recomputation; `theta-N-check.json` contains separately computed observations and gradient participation counts. `result.json` references these by hash, records scope, effective update hashes and any failure. Capture is synchronous and JSON retains raw tensor hex, so keep initial fixtures short. Setup/loading errors exit 2 before a run directory is created; errors during execution retain a result. An interruption may leave partial artifacts without a result and must not be treated as a pass.

`--updates 0` still performs real actor backward. `--updates 1` or `2` executes SGD on mean negative logprob of generated tokens, checks that parameter bytes actually changed, then performs a new comparison at each weight version. Sparse experts may have no gradient when not selected; participation counts do not claim full expert coverage. No reward function, optimizer parity, master-weight contract or resume is implemented here. Calibration updates modify only the in-memory model.

Results always have `certified: false` and `full_model_gpu_verified: false`: even a full checkpoint check is fixture-scoped, not a complete model/environment acceptance matrix. Do not feed this report to C0–C10 as if it were formal evidence. Native batching, cached PD, multi-device execution, cross-hardware equality, independent accuracy/gradient references, performance qualification and checkpoint resume are outside this contract.

## Python integration

```python
from rl_consistency.runtime.models import load_checkpoint
from rl_consistency.runtime.execution import SharedPrefixRuntime

# Set CUBLAS_WORKSPACE_CONFIG before process startup when using NVIDIA.
model, identity = load_checkpoint('/models/Qwen3-8B', 'qwen3',
                                  device='cuda:0', dtype='bfloat16')
runtime = SharedPrefixRuntime(model)
rollout = runtime.generate([{'id': 'sample', 'input_ids': [1, 2, 3]}], max_new_tokens=2)
checks = runtime.check(rollout)  # Independent old-logprob + actor backward; raises on mismatch.
# Actual actor gradients now reside on model parameters; check before an optimizer step.
```

`generate` alone has not performed parity verification; call `check` before accepting a result. `run_loaded` in `runtime/runner.py` combines these operations, writes reports and optionally performs calibration updates. It clears gradients on completion. It accepts an explicitly labeled tiny architecture for development; the CLI additionally validates the named full-model profile. Neither API should be called concurrently or while another process/thread mutates the model.

## Add a model

1. Extend `runtime/models.py` with the real config contract, official local checkpoint loader and pinned provider version. Refuse missing weights, unexpected quantization and unsupported routes. A profile label does not prove checkpoint provenance.
2. If the architecture uses the existing causal-forward interface, reuse `SharedPrefixRuntime`. For new stateful operators or output structures, add an adapter that explicitly defines reset semantics, observations and autograd behavior. Do not bypass state/routing capture to make a check pass.
3. Add tiny **real architecture** tests for FP32/BF16, local checkpoint roundtrip, independent generation/recompute/actor, effective updates, drift injection and failure handling. Tiny dimensions are development fixtures, distinct from the source-preserving proxies in C0–C10.
4. Run the actual full checkpoint and representative fixtures on the target GPU/software stack. Expand lengths, inputs, routing and state cases, then retain environment-bound evidence and separately measure accuracy and performance before advertising a supported release.

The initial registry covers Qwen3, Qwen3 MoE and the `qwen3_5` text path used by this repository's Qwen3.8-27B profile. The Hybrid loader uses the provider's official text-only checkpoint conversion; it does not load image/video inputs. The three architecture implementations share the executor, not model-specific copies of its comparison logic.

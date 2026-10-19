# RL-Consistency

An independent, model-extensible runtime and agent skill for bitwise train–inference consistency. **The default runtime requires no RL-Kernel installation.**

The shared executor runs generation, independent old-logprob recomputation and grad-enabled actor forward through the same trainable numerical path. Model adapters describe the architecture and load actual weights; the executor owns execution rules, comparison and failure handling.

**Status: reference runtime implemented; full-size GPU acceptance pending.** Installing a skill does not establish bitwise equality. A successful check establishes equality only for the recorded inputs, weights, environment and execution scope.

[中文说明](docs/README.zh-CN.md) · [Independent runtime guide](docs/independent-runtime.md) · [Architecture](docs/architecture.md)

## Model support and absolute consistency

| Model | Architecture | Independent execution | Automated verification completed | Full-model CUDA / ROCm bitwise acceptance |
|---|---|---|---|---|
| Qwen3-8B | Dense | Shared generation, recomputation and actor backward | Tiny architecture FP32/BF16 CPU parity; checkpoint load; two effective updates | Pending / Pending |
| Qwen3.8-27B | Hybrid Dense, text only; `qwen3_5` config family | Same executor, full-prefix GDN path | Tiny architecture FP32/BF16 CPU parity; nested checkpoint text load; two effective updates | Pending / Pending |
| Qwen3-30B-A3B | MoE | Same executor, independent router IDs/weights/logits capture | Tiny architecture FP32/BF16 CPU parity; checkpoint load; two effective updates | Pending / Pending |

Tiny tests instantiate the real Transformers architectures with small dimensions and random weights. They are tooling/integration tests, **not results for the named full checkpoints**. Model profile names are declarations; actual configs and weight file hashes are recorded. Qwen3.8-27B requires the repository's declared `qwen3_5` architecture contract; its name alone does not select an implementation.

## Run independently

Use an isolated environment with your platform's PyTorch already installed (ROCm PyTorch for AMD). The runtime extra does not install or replace PyTorch. The CPU integration baseline is PyTorch 2.6.0 and Transformers 5.20.1; GPU environments remain to be validated.

```bash
python -m pip install -e '.[runtime]'
rl-consistency runtime-models

rl-consistency runtime-check \
  --model Qwen/Qwen3-8B \
  --checkpoint /models/Qwen3-8B \
  --fixture examples/runtime-tokens.json \
  --device cuda:0 --dtype bfloat16 \
  --max-new-tokens 2 --output runs/qwen3-independent-001
```

`cuda:0` is also PyTorch's device spelling on ROCm. Supply an existing local safetensors checkpoint. For a natural-language input, replace `--fixture` with `--prompt 'Explain numerical reproducibility.'`; the checkpoint must include its tokenizer. The fixture contains synthetic token IDs, not a quality benchmark.

Change `--model`, `--checkpoint` and the output directory to select the other models. Add `--updates 2` to execute real SGD calibration updates and check theta0/theta1/theta2. These updates optimize token negative logprob in memory; they are not an RL reward run and do not save a trained checkpoint.

The command checks original-byte vocabulary logits and selected-token FP32 logprobs, decoder boundary hashes, and MoE router outputs. It retains actual rollout observations before independently recomputing them. Mismatches stop the run; output directories cannot be overwritten.

- Exit **0**: `equal_observed` for the executed fixture.
- Exit **1**: numerical mismatch; inspect `result.json` and the retained rollout.
- Exit **2**: unsupported route, invalid input, loading/execution error or incomplete check.

## Execution scope

The first independent provider, `shared-prefix-v1`, uses one device, physical batch size one, eager attention/experts, no KV cache and fresh full-prefix execution. All phases run in eval mode **with autograd enabled**; actor backward uses its independently computed graph. Outer request ordering cannot change the physical batch shape. Deterministic PyTorch algorithms are required; unsupported operations fail.

This is a slow reference path. It recomputes every prefix, copies observations to CPU and hashes model state. Eager attention and actor gradients consume substantial memory. A full 27B/30B model may exceed one GPU's capacity. There is no offload or distributed fallback. Use an isolated environment without `kernels`; Hybrid also rejects installed `fla`/`causal_conv1d`, which otherwise change provider dispatch.

The result does not establish native batching, cached prefill/decode parity, vLLM/Megatron parity, cross-device equality, independent mathematical accuracy, checkpoint resume or optimizer trajectory equality. It never automatically passes C0–C10 or marks a model certified. See [the runtime contract and extension steps](docs/independent-runtime.md).

## Add models

Add a profile and checkpoint adapter in `src/rl_consistency/runtime/models.py`, declare its semantic boundaries and routing/state observations, and exercise the real architecture through generation/recompute/backward and failure tests. Existing covered operators reuse the common executor. A new operator, quantization format or cache/state model requires provider work and fresh validation; adding a model name alone cannot grant support.

## Diagnostics and advanced engine validation

The [native first-divergence locator](docs/diagnostics.md) works before model adaptation: explicit PyTorch hooks, semantic DAG frontiers, independent repeats, callable replay and HTML comparison matrices. `diagnostic-demo` runs a synthetic CPU fault injection and intentionally exits 1.

The [C0–C10 evidence workflow](skills/train-inference-consistency/references/gates.md) remains available for broader model/engine acceptance. Read [evidence format](docs/evidence-format.md) and [workflow scope](skills/train-inference-consistency/references/workflow-scope.md). Its existing bundled vime recipes and `plan`/`preflight`/`run`/`rocm-check` commands are **optional legacy RL-Kernel integrations** with separate requirements. They are not prerequisites for `runtime-check`, and their evidence is not interchangeable with runtime-check reports.

## Skill and tests

Point your agent's skill loader at [`skills/train-inference-consistency/`](skills/train-inference-consistency/). Install the Python package first. The skill defaults to the independent runtime; it selects native diagnostics or advanced engine verification when requested.

```bash
python -m unittest discover -s tests -v
```

Without runtime dependencies, accelerator tests are skipped. The runtime CI job installs CPU PyTorch and the pinned Transformers version to execute all three tiny architectures. Evidence and local reviewer identities are not authenticated attestations; a signed release registry remains future work. [License](LICENSE).

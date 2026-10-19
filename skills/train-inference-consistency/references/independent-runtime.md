# Independent runtime procedure

Read `docs/independent-runtime.md` in the target RL-Consistency checkout for the executable contract and extension API. Inspect actual files and CLI help before running commands.

```bash
python -m pip install -e '.[runtime]'
rl-consistency runtime-models
rl-consistency runtime-check \
  --model Qwen/Qwen3-8B --checkpoint /models/Qwen3-8B \
  --fixture examples/runtime-tokens.json --device cuda:0 --dtype bfloat16 \
  --max-new-tokens 2 --output runs/independent-001
```

Replace paths using the user's real checkpoint. `--prompt TEXT` replaces the fixture and requires local tokenizer files. Other profile IDs are `Qwen/Qwen3.8-27B` and `Qwen/Qwen3-30B-A3B`. The Hybrid profile checks actual nested `qwen3_5` configuration; do not infer architecture or weight provenance from its name.

Requirements: platform PyTorch already installed, pinned Transformers 5.20.1, local safetensors, one sufficiently large CPU/GPU device, no `kernels`; Hybrid additionally excludes `fla` and `causal_conv1d`. Never uninstall packages from a shared environment without the user's applicable authorization. The dependency extra does not replace ROCm Torch. The tested CPU baseline is Torch 2.6.0; GPU/software combinations require fresh results.

This path generates greedy fixed-length tokens, independently recomputes old logprobs in reverse request order, independently runs actor forward and real backward, and checks decoder hashes, logits, logprobs and actual MoE routes. All phases use physical batch one, full prefixes, eval mode with autograd, eager kernels and no cache. Every prefix costs a complete model forward; gradients and eager attention increase memory. There is no automatic offload or multi-GPU support.

Use `--updates 2` only to request two effective in-memory SGD calibration steps and theta0/theta1/theta2 comparisons. The objective is mean negative logprob, not an RL reward; no trained checkpoint is saved. Different theta versions are never compared for equality.

Inspect `result.json`, referenced rollout/check files and errors. Exit 0 is fixture-scoped `equal_observed`; exit 1 is `mismatch`; exit 2 is an error. Setup failures may exit before creating the output directory. Missing results or partial artifacts are incomplete, never passing. Do not overwrite output directories or edit captured evidence. A differing decoder hash identifies an observed layer boundary; use native semantic diagnostics to refine it.

Reports deliberately remain `certified: false` and `full_model_gpu_verified: false`. They do not establish all-input correctness, native batching/PD, multi-device or cross-hardware equality, native vLLM/Megatron parity, independent gradient accuracy, optimizer trajectory, resume or performance. C0–C10 acceptance is separate. Tiny model fixtures cannot stand in for any named full-size model.

If the checkpoint/GPU is unavailable, finish supported CPU tooling and adapter work and state the missing run explicitly. Never download huge weights, synthesize a checkpoint under a real model's name, invent device results, or silently change the requested scope to obtain a pass.

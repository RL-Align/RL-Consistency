---
name: train-inference-consistency
description: Run independent bitwise train–inference checks with a shared trainable backend, extend model adapters, and locate first observed divergence. Use for standalone or out-of-the-box Qwen3-8B, Qwen3.8-27B Hybrid Dense and Qwen3-30B-A3B MoE execution without RL-Kernel, rollout versus actor logprob or routing drift, native semantic diagnostics, or advanced consistency evidence review.
---

# Train–Inference Consistency

Use RL-Consistency as an independent shared execution backend with model adapters and evidence tools. The default route requires PyTorch and pinned Transformers, not RL-Kernel. Never infer model certification from an installed skill, a successful command, a repository directory, or a `status: verified` field.

Inspect the actual checkpoint, hardware, installed packages and requested equality scope. Select the independent runtime for standalone use; select native diagnostics for existing-engine drift; select workflow v2 only for explicit broader engine acceptance.

## Independent execution (default)

Read [the independent runtime procedure](references/independent-runtime.md).

1. Inspect `docs/independent-runtime.md`, `runtime-models`, the actual config and available device memory. Use the user's local checkpoint. Do not create random weights or silently substitute a smaller model.
2. Use an isolated environment with platform PyTorch and `pip install -e '.[runtime]'`. Do not install RL-Kernel, vLLM, Megatron, FLA or hub kernels for this route. Do not alter a shared environment to remove packages; select a dedicated environment when providers conflict.
3. Run `runtime-check` with `--model`, `--checkpoint`, a token fixture or prompt, explicit device/dtype and a new output directory. On AMD use PyTorch's `cuda:0` spelling. Start with short inputs; this path needs one device and full-prefix eager execution plus actor gradients.
4. Inspect the exit code and `result.json`. Exit 0 means equality observed only for its fixture/scope; 1 means mismatch; 2 means error/unsupported/incomplete. Preserve failures. For mismatch inspect the first observed decoder boundary and use semantic diagnostics for finer localization.
5. When effective-update checks are requested, use `--updates 2`; inspect the two changed state hashes and three separate same-weight comparisons. Report these as SGD negative-logprob calibration, not a reward-training run, optimizer parity or checkpoint resume.
6. Report measured scope and limitations. All three full-size GPU model acceptance statuses remain pending until actual evidence is reviewed. Tiny CPU architecture tests do not establish Qwen full-model GPU parity. Never map this report into C0–C10 or claim native batching, cached PD, vLLM/Megatron, multi-device or cross-hardware equality.
7. For a new model, extend its profile/loader and explicit state/routing observations, reuse covered operators, add real architecture/failure tests, then run the actual checkpoint. New operators or execution modes require provider development; adding a model name cannot grant support.

## Native diagnosis or advanced engine workflow

For an unregistered model or a request to locate native divergence, start with
[standalone diagnostics](references/diagnostics.md). Inspect both native engines,
map semantic boundaries, capture independent repeats, generate the DAG matrix
and replay candidates. This route does not require `prepare-model`, a recipe or
RL-Kernel operators. Return to the workflow below when developing and verifying
the strict runtime; a diagnostic match cannot certify model support.

1. Read the target repository's instructions and `docs/architecture.md`. Inspect the actual model, engine versions, hardware and user-requested comparison scope.
   For Qwen3.8-27B, read [the Hybrid model guide](references/qwen38-27b.md) and
   run `prepare-model` using the real HF config before binding a recipe. Preserve
   nested text config and L/L/L/F order; use an 8-layer proxy by default. This
   adapter prepares a BF16 text-only investigation; it does not launch or certify it.
   For Qwen3-30B-A3B, read [the MoE guide](references/qwen3-30b-a3b.md), then
   use `prepare-model` with the actual HF config. Keep 128 experts/top-8 in the
   default 4-layer proxy. Capture independent routing alongside outputs; R3
   replay is a separate diagnostic and cannot pass independent-routing gates.
2. Run `rl-consistency list`, export a matching candidate with `export-recipe`, then run `check`. Pin real model/fixture hashes, source commits, Python package lock and machine profile. Do not fill missing identities with plausible values.
3. Read [gates.md](references/gates.md) and [numerical-contract.md](references/numerical-contract.md). Obtain an explicit scope: same backend, cross topology, cross hardware, or optimizer trajectory. The current implementation supports within-backend train/rollout evidence only; do not relabel that as the other scopes.
4. Initialize a new run with a fully bound recipe. Keep run inputs immutable. A changed model, numerical rule, source, environment, scope or tolerance requires a new run.

## Execute the advanced workflow

- Use `scripts/run_gate.py` to submit real evidence to C0–C10. Read `docs/evidence-format.md` before writing a producer. The CLI checks artifacts and records verdicts; never write successful result files manually.
- Follow C0 → C2 → C3 → C4 → C5 → C6 → C7 → C8 → C9. Develop C1 independently after C0. Join both branches at C10. Native tuning and consistency work may proceed in isolated agent worktrees, with separate output directories and agreed measurement inputs.
- Delegate one agent to native baseline tuning and another to consistency work when the host supports agents and the user has authorized that work. Share fixed workload requirements, not writable baseline measurements. If delegation is unavailable, run the roles sequentially with the same isolation.
- Read [baseline-policy.md](references/baseline-policy.md) before C1/C10. Freeze the baseline after its human review. Do not slow the native path, cherry-pick runs, change work, or hide failures to improve the comparison.
- Construct a reduced model by selecting source layers, preserving operator dimensions, quantization and relevant layer transitions. Usually 4–6 layers suffice; increase the count when coverage demands it. Do not weaken the bound full-model specification to create a small model.
- At C5 compare two actual paths separately: native old-logprob recomputation and the grad-enabled actor forward, each against rollout at the same weight version. Record model/grad modes, output requires_grad and executed callsite. Reusing rollout values or substituting a diagnostic forward is not verification.
- Read [debugging.md](references/debugging.md) and [standalone diagnostics](references/diagnostics.md) for a mismatch. Preserve failing inputs, find the earliest observed divergent frontier in the declared dependency graph, then replay and verify the cause. Missing ancestors and opaque interiors limit localization. Do not relax bitwise acceptance, change the baseline, or stop at matching generated tokens.
- Run independent mathematical forward and gradient references. A reference with the same fixed reduction tree checks consistency only; it is not sufficient evidence of numerical accuracy.
- Complete theta0 compare → effective update → theta1 compare → effective update → theta2 compare, with normal exit. Check finite/nonzero gradients and complete master parameter change; deployment bytes may stay unchanged after quantization. Keep every comparison within one weight version.
- Validate the full model, declared context/topology matrix, and restored checkpoint's first forward/update. Preserve model, optimizer, scheduler, RNG and data cursor state. Saving or loading alone does not pass.
- Record the exact source/binary artifacts loaded by each worker and the actual operator routes. Reject unsupported routes and fallback. Do not substitute a source checkout SHA for loaded-worker provenance.

## Human decisions

Follow [release-policy.md](references/release-policy.md). Human baseline, precision, pre-C9 recipe and final release reviews remain explicit. The `review` command records an existing decision; it does not authenticate a person. Do not invent a reviewer or record an approval that the user has not given. Continue all authorized development and diagnostics while a review is pending.

## Report

Use `scripts/summarize_run.py` to create HTML and inspect `status`. Report the declared scope, passed/failed/missing gates, measured accuracy, performance overhead, evidence limitations and next actionable task. If there is no compatible accelerator or model checkpoint, complete CPU tooling checks and report GPU gates as pending. Do not certify a candidate or claim reward gains from CPU tests.

## Optional compatibility boundary

Use `rocm-plan`, `rocm-check` and the Qwen3-8B `run` bridge only for an explicitly requested existing RL-Kernel/vime integration; read `docs/rocm-quickstart.md`. These commands have separate external dependencies and do not pass C5/C8/C9 automatically. Existing `inspect-model`/`prepare-model` commands and bundled vime recipes remain preparation tools for the advanced workflow. They are not prerequisites for the independent runtime. Hybrid/MoE native dual-engine providers, full evidence producers, proxy weight materialization, distributed/cache execution and a signed registry remain extension work. Never replace missing components with dummy success outputs.

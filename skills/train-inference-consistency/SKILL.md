---
name: train-inference-consistency
description: Develop and audit bitwise train–inference consistency using RL-Kernel. Use to onboard models including Qwen3-8B, Qwen3.8-27B Hybrid Dense and Qwen3-30B-A3B MoE, diagnose rollout versus training logprob or routing drift, validate batch or prefill/decode invariance, compare a strict path with a native baseline, or review verification evidence.
---

# Train–Inference Consistency

Treat RL-Consistency as an evidence workflow and RL-Kernel as the numerical implementation. Never infer model certification from an installed skill, a successful launcher, a repository directory, or a `status: verified` field.

Use workflow v2 and read [the workflow scope](references/workflow-scope.md).
Inspect the user's framework, hardware and numerical requirements before
selecting providers, launch settings or benchmarks.

## Begin

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

## Execute the workflow

- Use `scripts/run_gate.py` to submit real evidence to C0–C10. Read `docs/evidence-format.md` before writing a producer. The CLI checks artifacts and records verdicts; never write successful result files manually.
- Follow C0 → C2 → C3 → C4 → C5 → C6 → C7 → C8 → C9. Develop C1 independently after C0. Join both branches at C10. Native tuning and consistency work may proceed in isolated agent worktrees, with separate output directories and agreed measurement inputs.
- Delegate one agent to native baseline tuning and another to consistency work when the host supports agents and the user has authorized that work. Share fixed workload requirements, not writable baseline measurements. If delegation is unavailable, run the roles sequentially with the same isolation.
- Read [baseline-policy.md](references/baseline-policy.md) before C1/C10. Freeze the baseline after its human review. Do not slow the native path, cherry-pick runs, change work, or hide failures to improve the comparison.
- Construct a reduced model by selecting source layers, preserving operator dimensions, quantization and relevant layer transitions. Usually 4–6 layers suffice; increase the count when coverage demands it. Do not weaken RL-Kernel's existing full-model specification to create a small model.
- At C5 compare two actual paths separately: native old-logprob recomputation and the grad-enabled actor forward, each against rollout at the same weight version. Record model/grad modes, output requires_grad and executed callsite. Reusing rollout values or substituting a diagnostic forward is not verification.
- Read [debugging.md](references/debugging.md) for a mismatch. Preserve the failing inputs, find the earliest differing intermediate in declared execution order, and correct the first cause. Do not relax bitwise acceptance, change the baseline, or stop at matching generated tokens.
- Run independent mathematical forward and gradient references. A reference with the same fixed reduction tree checks consistency only; it is not sufficient evidence of numerical accuracy.
- Complete theta0 compare → effective update → theta1 compare → effective update → theta2 compare, with normal exit. Check finite/nonzero gradients and complete master parameter change; deployment bytes may stay unchanged after quantization. Keep every comparison within one weight version.
- Validate the full model, declared context/topology matrix, and restored checkpoint's first forward/update. Preserve model, optimizer, scheduler, RNG and data cursor state. Saving or loading alone does not pass.
- Record the exact source/binary artifacts loaded by each worker and the actual operator routes. Reject unsupported routes and fallback. Do not substitute a source checkout SHA for loaded-worker provenance.

## Human decisions

Follow [release-policy.md](references/release-policy.md). Human baseline, precision, pre-C9 recipe and final release reviews remain explicit. The `review` command records an existing decision; it does not authenticate a person. Do not invent a reviewer or record an approval that the user has not given. Continue all authorized development and diagnostics while a review is pending.

## Report

Use `scripts/summarize_run.py` to create HTML and inspect `status`. Report the declared scope, passed/failed/missing gates, measured accuracy, performance overhead, evidence limitations and next actionable task. If there is no compatible accelerator or model checkpoint, complete CPU tooling checks and report GPU gates as pending. Do not certify a candidate or claim reward gains from CPU tests.

## Current integration boundary

Use `rocm-plan` and `rocm-check` with `docs/rocm-quickstart.md` for an explicitly requested Qwen3-8B ROCm fixed-policy diagnostic. It retains the real upstream sidecars, validates rank/token coverage and reports a scoped raw-byte verdict; it does not pass C5 or C8/C9. The Qwen3-8B `run` bridge remains available for reviewed experiments. For Qwen3.8-27B and Qwen3-30B-A3B, use `inspect-model`/`prepare-model` with the linked model guides (`docs/qwen38.md`, `docs/qwen3-moe.md`); continue provider and capture development rather than relabeling the Qwen3-8B launcher. Full dual-path producers, Hybrid/MoE strict runtime, proxy weight materialization and a signed recipe registry remain extension work. Never replace missing components with dummy successful outputs.

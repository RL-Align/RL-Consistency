---
name: train-inference-consistency
description: Develop and audit bitwise train–inference consistency using RL-Kernel. Use to onboard a model, diagnose rollout versus training logprob drift, validate batch or prefill/decode invariance, compare a strict path with a native baseline, or review a recipe's verification evidence.
---

# Train–Inference Consistency

Treat RL-Consistency as an evidence workflow and RL-Kernel as the numerical implementation. Never infer model certification from an installed skill, a successful launcher, a repository directory, or a `status: verified` field.

## Begin

1. Read the target repository's instructions and `docs/architecture.md`. Inspect the actual model, engine versions, hardware and user-requested comparison scope.
2. Run `rl-consistency list`, export a matching candidate with `export-recipe`, then run `check`. Pin real model/fixture hashes, source commits, Python package lock and machine profile. Do not fill missing identities with plausible values.
3. Read [gates.md](references/gates.md) and [numerical-contract.md](references/numerical-contract.md). Obtain an explicit scope: same backend, cross topology, cross hardware, or optimizer trajectory. The current implementation supports within-backend train/rollout evidence only; do not relabel that as the other scopes.
4. Initialize a new run with a fully bound recipe. Keep run inputs immutable. A changed model, numerical rule, source, environment, scope or tolerance requires a new run.

## Execute the workflow

- Use `scripts/run_gate.py` to submit real evidence to C0–C10. Read `docs/evidence-format.md` before writing a producer. The CLI checks artifacts and records verdicts; never write successful result files manually.
- Follow C0 → C2 → C3 → C4 → C5 → C6 → C7 → C8 → C9. Develop C1 independently after C0. Join both branches at C10. Native tuning and consistency work may proceed in isolated agent worktrees, with separate output directories and agreed measurement inputs.
- Delegate one agent to native baseline tuning and another to consistency work when the host supports agents and the user has authorized that work. Share fixed workload requirements, not writable baseline measurements. If delegation is unavailable, run the roles sequentially with the same isolation.
- Read [baseline-policy.md](references/baseline-policy.md) before C1/C10. Freeze the baseline after its human review. Do not slow the native path, cherry-pick runs, change work, or hide failures to improve the comparison.
- Construct a reduced model by selecting source layers, preserving operator dimensions, quantization and relevant layer transitions. Usually 4–6 layers suffice; increase the count when coverage demands it. Do not weaken RL-Kernel's existing full-model specification to create a small model.
- Recompute training logprobs independently from rollout tokens at the same weight version. Reusing rollout logprob values is not verification. Match masks, positions, sampling/support semantics and logical ordering first.
- Read [debugging.md](references/debugging.md) for a mismatch. Preserve the failing inputs, find the earliest differing intermediate in declared execution order, and correct the first cause. Do not relax bitwise acceptance, change the baseline, or stop at matching generated tokens.
- Run independent mathematical forward and gradient references. A reference with the same fixed reduction tree checks consistency only; it is not sufficient evidence of numerical accuracy.
- Perform at least two real optimizer updates. Compare both sides again after each weight synchronization. Keep each comparison within one weight version. Never compare stale rollout scores against an updated policy and call the expected difference a consistency failure.
- Validate the full model, declared context/topology matrix, and restored checkpoint's first forward/update. Preserve model, optimizer, scheduler, RNG and data cursor state. Saving or loading alone does not pass.
- Record the exact source/binary artifacts loaded by each worker and the actual operator routes. Reject unsupported routes and fallback. Do not substitute a source checkout SHA for loaded-worker provenance.

## Human decisions

Follow [release-policy.md](references/release-policy.md). Human baseline, precision and final recipe reviews remain explicit. The `review` command records an existing decision; it does not authenticate a person. Do not invent a reviewer or record an approval that the user has not given. Continue all authorized development and diagnostics while a review is pending.

## Report

Use `scripts/summarize_run.py` to create HTML and inspect `status`. Report the declared scope, passed/failed/missing gates, measured accuracy, performance overhead, evidence limitations and next actionable task. If there is no compatible accelerator or model checkpoint, complete CPU tooling checks and report GPU gates as pending. Do not certify a candidate or claim reward gains from CPU tests.

## Current integration boundary

The Qwen3 bridge generates and executes the pinned RL-Kernel `run` command after host preflight. Native engine trace producers, generalized model adapters, automatic reduced-model construction and a hosted verified-recipe registry remain extension work. Use the existing upstream adapters and implement concrete producers in the actual engine workers; never replace these missing components with dummy successful outputs.

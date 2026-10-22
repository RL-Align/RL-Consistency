# C10 — Deliver reproducible implementation and measured results

**Entry:** valid C1 frozen baseline and C9 full-model acceptance, with all required prior stages and decisions. Read the run ledger and inspect referenced artifacts before composing a release claim.

## Assemble and verify

1. Deliver target-repository changes with exact refs/build instructions, dependency/environment identities, model/fixture acquisition and verification steps, explicit supported dispatch rules and actual executable reproduction commands. Keep user-specific secrets and machine credentials out of the bundle.
2. Provide the resolved numerical/model contract, operator/backward map, full test matrix, raw evidence index/digests, independent references, retained failures/fixes and required review records. Ensure another agent can trace every claimed pass to actual artifacts.
3. Reproduce the delivery entrypoint on the declared environment and verify it selects the accepted providers and catches an unsupported configuration. A different fallback path requires separate validation, not reuse of the strict result.
4. Benchmark strict and frozen native configurations at equal work using C1's policy and identical timing/memory boundaries. Report measured raw data, warmup, repeats, variability, throughput/latency/step cost, memory and consistency behavior. Separate free-running RL quality/reward trajectories from equal-work performance.
5. Explain residual overhead with actual profiles. Do not promise faster execution or improved reward without measured evidence. Any optimization after acceptance reopens affected equality/accuracy/full-model gates.
6. Complete [release-report.md](../templates/release-report.md), supported-range/model table and handoff. Record the actual release decision under the target project's policy. Pending decisions or missing evidence remain visible.

## Exit

Pass only when implementation, reproduction, model/environment-bound consistency/accuracy evidence and fair performance results are complete for the advertised scope, with required decisions recorded. A Markdown guide, self-reported status, successful launcher or CPU fixture does not establish a verified model release.

If only a development subset was requested or achievable, deliver that useful subset with passed/failed/blocked/stale gates and concrete next actions. Do not label it C0–C10 complete. State model revision, engines, hardware/software, dtype, parallelism, batch/context/cache ranges and exclusions beside every verified claim. Do not turn within-backend train/rollout parity into cross-hardware or optimizer-trajectory claims.

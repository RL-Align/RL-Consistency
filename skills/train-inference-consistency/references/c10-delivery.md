# C10 — Deliver reproducible consistency

**Entry:** C9 full-model acceptance and all required correctness stages/decisions. In default `consistency-only` mode C1/C4 are `not-required`; no native baseline, profile, timing or speed target is a prerequisite. When performance is explicitly requested, also require C1/C4 and the agreed performance results. Inspect raw artifacts before composing a release claim.

## Assemble and verify

1. Deliver target-repository changes with exact refs/build instructions, dependency/environment identities, model/fixture acquisition and verification steps, supported dispatch rules and actual executable reproduction commands. Keep secrets and credentials out of the bundle.
2. Provide the numerical/model contract, operator/backward map, full test matrix, raw evidence index/digests, independent references, retained failures/fixes and applicable review records. Ensure another agent can trace every claimed pass to actual artifacts.
3. Reproduce the delivery entrypoint on the declared environment and verify it selects the accepted providers and catches an unsupported configuration. A different fallback path requires separate validation. A common slow implementation is acceptable when the actual training/rollout branches independently execute it and complete all required checks.
4. Complete [release-report.md](../templates/release-report.md), the model/supported-range table and handoff. State the acceptance mode. For consistency-only delivery write performance `not evaluated`; describe deliberate serialization or other choices needed to reproduce the path, without creating a performance gate.
5. Only when performance work is requested, compare strict and frozen native configurations at equal work using C1's policy. Report raw timings, warmup, repeats, variability and identical memory semantics; explain overhead with measured profiles. Never infer speed or reward improvement from equality.
6. Record the actual release decision under the target project's policy. Any later optimization that changes execution reopens affected equality, accuracy and full-model gates. Required pending decisions or missing correctness evidence remain visible.

## Exit

Pass when implementation, reproduction and model/environment-bound consistency/accuracy evidence are complete for the advertised scope, with applicable decisions recorded. In consistency-only mode, a slow implementation and absent performance measurements are not blockers. In performance mode, complete its additional agreed requirements before claiming that mode passed.

Report `C0–C10 consistency-only acceptance complete; C1/C4 not-required; performance not evaluated` only after all required correctness stages pass. Do not say all eleven gates passed. A Markdown guide, self-reported status, successful launcher or CPU fixture does not establish a verified model release.

If only a development subset was requested or achievable, deliver that subset with actual statuses and next actions. State model revision, engines, hardware/software, dtype, parallelism, batch/context/cache ranges and exclusions beside every verified claim. Do not turn within-backend train/rollout parity into cross-hardware or optimizer-trajectory claims.

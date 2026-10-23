# C1 — Establish a strong native baseline

**Applicability:** optional performance track. In default `consistency-only` mode record C1 as `not-required`, reason `performance not requested`, and proceed without native tuning or benchmarks. This does not waive C7 independent mathematical references or required model identity inspection.

**Entry when enabled:** C0 and a defined target workload. Keep native and strict work isolated by worktree/config/output directory.

## Implement and measure

1. Reproduce the native framework configuration the user would reasonably deploy. Confirm actual model/precision, topology, batching, caching, parallelism and scoring/task semantics. Record sources, loaded binaries, inputs, output work and hardware allocation.
2. Tune the native path independently using its legitimate optimizations. Preserve all trials, parameters, failures and profiling observations. Do not disable native caching/graphs/fusion or serialize work solely to make the consistency path look faster.
3. Implement a reproducible equal-work benchmark in the target project. Control prompt/response tokens, measured step count, hardware allocation and timing boundaries. If using replayed token work, label it as such and retain the replay fixture; separately report free-running RL lengths/rewards.
4. Separate compile/load/warmup from steady-state measurements. Include the same forward/backward/optimizer/synchronization/communication boundaries on both sides. Retain at least three measured steps after warmup; use independent repeated runs and uncertainty estimates for performance claims.
5. Measure memory using identical definitions, such as peak allocated memory over the same phase; distinguish allocator/reserved/device-level measurements. Record outliers, synchronization policy and contention. Do not compare unsynchronized GPU timings.
6. Freeze the selected native configuration, workload and raw results with their digests and actual baseline decision. Continue other authorized development while any required owner review is pending; do not invent it.

## Artifacts and exit

Retain benchmark commands, resolved config, environment/source identities, native tuning trials, raw per-step timings, warmup policy, output-token/work counts, memory data, profiles and the freeze/review record. Pass when the baseline is reproducible, reasonably tuned for the declared workload and accepted under the target project's review policy.

A mismatching native path can still be the native baseline; report its consistency behavior separately. Do not fix the native reference by secretly enabling strict providers. A changed workload or baseline revision needs new measurements and a new freeze record. When reporting performance at C10, use this frozen baseline, not a cherry-picked trial.

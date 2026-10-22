# C4 — Remove structural performance defects

**Entry:** C3 inference path and retained consistency fixtures. This gate inspects implementation structure; it is not independent accuracy (C7) or final performance acceptance (C10).

## Profile and repair

1. Profile the actual proxy workload and loaded execution route. Record real operator shapes, call counts, timings, allocation/workspace traffic, launch/synchronization behavior and communication. Keep raw profiler artifacts with hashes and the exact capture command.
2. Inspect excessive per-token/per-expert host loops, redundant projections/casts/layout conversions, repeated weight preparation, quadratic recomputation where unintended, device-host copies, workspace churn, cache updates and communication serialization. Distinguish intentional contract costs from avoidable implementation defects.
3. Inspect Graph/capture behavior when in scope: stable buffers, prepared weights, allocation rules, launch compatibility and actual captured kernels. A Graph flag without exercised execution proves nothing.
4. Implement a targeted structural fix in the target project. Preserve masks, logical ordering, precision, reductions, state and backward requirements. Do not change the workload or disable native baseline optimizations to improve a comparison.
5. Rerun impacted C3 cases and compare raw profiles before/after. If the numerical path changes, update its identity/contract and mark affected downstream results stale. Measure observer overhead separately.
6. Record remaining bottlenecks and a concrete training-path budget for C6, tied to the native workload. Any deliberately slower reference path must remain labeled with its limited purpose/scope.

## Artifacts and exit

Retain actual shapes/call counts, raw profiles, source changes, before/after measurements, Graph/workspace checks, consistency reruns and residual-cost explanation. Pass when structural defects in the intended path are resolved or explicitly accepted as scoped design tradeoffs with measured cost. A graph of synthetic timings or a list of intended kernels is insufficient.

If a repair breaks equality, return to C3 and diagnose it. Do not continue claiming the prior path's consistency after changing fusion, launch geometry or reduction order. If the intended performance budget is infeasible, surface the measured tradeoff and required decision rather than hiding it.

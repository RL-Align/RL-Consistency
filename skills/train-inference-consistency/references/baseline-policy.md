# Baseline and performance

Use separate worktrees and output directories for native tuning and the strict implementation. Tune the native implementation independently, retain raw trials and obtain a human decision before freezing it. Shared inputs include model, task, hardware allocation, topology, sampling semantics, output work and timing boundaries.

The first toolkit comparator uses controlled, replayed token work: at least three measured end-to-end steps after warmup, equal measured step count and output-token count, with matching settings and fixtures. Keep compile/load time separate and state whether graph warmup, communication and synchronization are included. Record peak allocated-memory measurement semantics identically.

Free-running RL trajectories can produce different lengths/rewards. Report them separately; do not compare their raw step times as equal-work benchmarks. For statistically meaningful performance claims add repeated runs and uncertainty estimates; the initial comparator reports a mean only.

Optimize one bottleneck at a time. Re-run impacted consistency/accuracy gates and full acceptance after code changes. A new source/build fingerprint requires a new bound recipe/run; a prior release decision does not authorize a new numerical path.

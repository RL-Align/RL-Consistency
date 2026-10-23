# Locate first divergence before model adaptation

Use this procedure on existing native paths even when no model guide or strict provider exists. Implement the necessary capture adapters in the target project. No registry entry, special CLI or kernel replacement is required to begin.

## Establish a comparable failing case

1. Inspect actual rollout, independent old-logprob and actor execution. Freeze weights, exact token IDs/masks/positions, scoring semantics and necessary state. Preserve the original failing request and worker provenance.
2. Inventory module boundaries, then map them to semantic nodes and dependencies. Record inputs/outputs and known state for each node. Module names/shapes alone do not prove equivalence. Treat fused/opaque interiors as explicit intervals.
3. Predeclare expected requests/tokens/nodes/ranks and record what cannot be observed. Capture each native side twice from equivalent reset state into four independent attempt directories. Check same-side repeatability before attributing stable cross-path differences.
4. Compare logically aligned raw bytes and coverage using [evidence.md](evidence.md). Keep rollout/old and rollout/actor cases separate. If weights, semantics, inputs or state differ, resolve that before blaming arithmetic.

## Map scheduling and repeated module calls explicitly

One external batch need not correspond to one internal forward. An asynchronous engine can schedule different request subsets in the first forward of two otherwise identical runs. Capture the actual request/token/position-to-row mapping; label unmatched scopes `incomparable`, rather than treating different physical shapes as numerical instability or silently dropping rows. Token values alone are not a general identity: repeated tokens, duplicate prompts and reused positions require explicit request and logical-position keys.

Internal worker request IDs may differ from public response IDs. Record their association at the actual execution boundary; do not invent a suffix-stripping rule after comparison fails. Reused module objects (for example a shared RoPE instance) also require logical layer/call-site or invocation identity. A module-name-only dictionary can overwrite several executions and fabricate coverage.

Hooks that copy GPU tensors to the CPU can change scheduling. Record this instrumentation scope and preserve physical layouts. Use identical saved inputs for repeated actual-provider replay to isolate arithmetic, then rerun the original failing path with the targeted repair. An incomplete first-forward capture cannot certify every request or decode step.

## Follow the semantic DAG

For every node whose outputs differ, inspect its input ports, state, repeat stability and predecessor coverage. A node with matching incoming values/state but differing outputs is a useful local candidate. If inputs already differ, trace its predecessors first. If an ancestor is missing or opaque, report the smallest observed interval and unresolved coverage.

There can be multiple parallel divergent frontiers. Floating-point differences can disappear and later reappear; equal final outputs do not rule out intermediate drift. Do not use binary search assuming monotonic propagation, lexicographic layer order or cross-device wall clocks as causal order.

Use a matrix of semantic nodes versus test cases, with `equal`, `different`, `missing`, `incomparable` or `unstable` cells. Link every cell to its captures and repeat verdicts. Add finer boundaries only where needed. An HTML viewer is optional target-project tooling; the matrix and evidence must remain inspectable without it.

## Replay and establish cause

Replay each candidate with identical original input bytes and equivalent complete state using the actual providers. Preserve physical shape/layout, masking, precision, kernel selection, graph/cache mode and relevant communication context. Repeat each replay independently. A standalone callable mismatch establishes a local difference in that replay context; confirm the diagnosis on the original execution path before declaring the production cause.

Inspect GEMM/reduction partitions, accumulator type, cast/fusion/FMA points, transcendental functions, attention/KV metadata, quantization scales and MoE decisions. Compare fixed-order contract references and independent high-precision math separately. For communication, check both endpoints' dtype/shape/byte count, offsets, logical order, sequence and group mapping, including uneven or empty payloads when in scope.

Apply one targeted change, preserve a regression fixture and rerun the original failing case plus affected invariance/accuracy cases. If the failure survives, retain the attempted fix's evidence and continue from the observed frontier. Do not replace real missing data with synthetic equal values.

## Report the limits

Record first observed frontier(s), upstream gaps, opaque regions, input/state equality, repeat stability, replay context/result, confirmed cause if established, code change and regression results. Partial module state is not complete runtime state. A matched final logprob, successful local replay or an instrumented reference route cannot certify the unobserved native production execution.

For an unknown model, complete inventory and boundary mapping first, then use [model-extension.md](model-extension.md) to develop the strict implementation. Preserve a blocked status where the actual engine cannot expose required evidence.

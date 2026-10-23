# Build consistency without a performance target

Use this route by default. The goal is a working, mathematically valid training–inference implementation with original-byte equality in its declared scope. There is no throughput, latency, speedup or native-cost threshold. C1 native tuning and C4 structural optimization are optional; do not spend time on them unless requested. A slow implementation is a valid final delivery after the required correctness gates pass.

## Choose an executable common arithmetic path

1. Inspect the real checkpoint, objective, framework entrypoints and available devices at C0. Reuse compatible operations where their exercised behavior meets the contract. Where implementations disagree, prefer a simple common implementation over reconciling two complex optimized kernels.
2. Write that implementation in the target project and bind it to rollout, independent old-logprob recomputation and the grad-enabled actor. The three branches may share source code, operator providers and mathematical algorithms. Execute fresh computations with equivalent weights and state; sharing an implementation does not permit sharing its computed scores or replaying routing decisions.
3. Disable optional fusion, autotuning, graph compilation or specialized dispatch where they prevent the intended arithmetic from being controlled. Use explicit operations, fixed shapes/order, a straightforward attention path, sequential recurrence or serialized communication when useful. No separate permission is needed just because these choices are slow.
4. Preserve explicitly required engines, hardware, precision, batching, cache/state behavior, topology and objective. If these are unspecified, select and record a tractable initial configuration and its limits. A common uncached path can be the delivered implementation when caching is not required; it does not establish cached decode parity. Do not turn permission to ignore speed into permission to omit a required model path or distributed configuration.
5. Integrate one operator family at a time. Check its byte equality, independent mathematical accuracy and usable gradients, then assemble the representative proxy. Follow C3→C5→C6→C7→C8→C9; do not stop at the proxy or a synthetic model.

## Concrete implementations to develop

These are construction choices, not pre-verified providers. Inspect emitted/executed arithmetic and validate each choice on the target environment.

| Component | Simple implementation to try first | Required check |
|---|---|---|
| Linear/GEMM | Process each logical token row with the same operand shapes. Use a fixed K partition and ordered accumulation; if a backend changes reduction arithmetic, implement a fixed-order dot/reduction or scalar fallback with explicit accumulation and cast points | Batch/neighbour invariance, projection bytes and C7 math/gradients; a fixed input shape alone is not proof |
| Norm, softmax, logprob | Use one explicitly ordered reduction over valid coordinates, with fixed accumulator precision, epsilon placement, normalization and casts | Same valid vocabulary/feature axis, signed-zero handling, finite outputs and independent accuracy |
| Elementwise, RoPE, residual | Use common formulas, constants/transcendental providers, layouts and cast/fusion boundaries on all branches | Identical logical positions and intermediate bytes; do not assume all hardware transcendental implementations match |
| Causal attention | Compute each query from its valid causal keys in logical key order; use common QK, max/exp/sum/divide and AV operations. Avoid shape-dependent split-KV or fused alternatives until validated | Each token sees the same prefix/mask/positions; cached K/V precision must match if caching is required |
| Hybrid/stateful layers | Use the same sequential convolution/state recurrence for prefill, decode and actor, preserving the actor's computation graph. Acceleration by parallel scan/chunk formulas is optional | Equivalent initial state, transitions, resets, request ownership, final states and saved values for backward |
| MoE | Recompute router outputs independently; use an explicit tie rule, stable token/slot mapping, sequential expert evaluation and ordered weighted combine | Preserve all original experts/top-k/capacity semantics and gradients; no forced/replayed routes as acceptance |
| Distributed reductions | Gather values and combine in an explicit logical order, or implement canonical logical chunks with a fixed combine order. Serialize communication if needed | Cross-topology claims also require topology-independent local partial sums; rank order alone is insufficient |
| Actor/backward | Use ordinary autograd through explicit differentiable operations when valid; otherwise implement and independently verify custom backward | No detach/no-grad surrogate, all expected gradients, actual objective and optimizer semantics |

For example, a simple dot-product contract can specify accumulator zero, increasing logical `k`, multiplication rounding, addition rounding, then one final output cast. Implement those decisions explicitly. State whether FMA is permitted; a compiler-contracted multiply/add is not equivalent to two prescribed roundings. Apply the same contract to every branch. A different contract may be more accurate; validate accuracy at C7 instead of assuming sequential arithmetic is always adequate.

For batch invariance, unpack requests by logical identity, run the same per-request/per-token arithmetic, and restore the requested output layout. This can preserve the external batch interface while serializing internal computation. It must preserve causal masks, loss normalization and the target model's intended semantics; it is unsuitable if it silently removes genuine cross-sample operations. Materialize outputs without breaking the actor graph. Test the original batch/permutation cases rather than reporting only a physical batch-one experiment.

For attention, do not include future/padding coordinates merely to match physical tensor dimensions. Iterate the same valid prefix for a token in both scoring and generation. For recurrent models, an identical recurrence evaluated from equivalent state is the starting point; copying rollout's hidden states or routes into actor is not an independent training computation.

## Verify accumulation precision in the differentiable implementation

A per-token forward written as a list of ordinary linear calls can be byte-stable while backward accumulates separate contributions into a BF16 parameter gradient. The repeated BF16 rounding may violate a predeclared accuracy threshold even when forward and input gradients pass. Inspect the actual accumulation dtype and cast points; do not assume ordinary autograd implies sufficiently accurate parameter gradients.

If the measurement requires a correction, accumulate parameter-gradient contributions in the declared higher precision and logical order, with an explicit final cast. A custom backward must use the real saved inputs and incoming derivatives, preserve all expected input/weight/bias gradients and be checked against independent mathematical formulas. Keep the failed case and original thresholds. Passing this local check does not establish optimizer accuracy, distributed normalization, higher-order gradients or the full model's C6/C7 gates.

## Finish the real task

Run rollout→old and rollout→actor comparisons separately at theta0, theta1 and theta2, with two effective optimizer updates, synchronized weights and invalidated/rebuilt dependent state. Compare only within a weight version. Keep old policy scores frozen when the actual RL objective requires that; use fresh synchronized cases for the new-version comparisons.

Pass C7 independently of equality, then complete the real full model and resumed first forward/update at C9. Implement explicit guards for unsupported configurations. Record C1/C4 as `not-required`, not `pass`, and C10 as a consistency-only delivery with performance `not evaluated`.

A slow run that completes can pass. A run that cannot load, exhausts memory, hangs or has not finished leaves the affected correctness checks blocked or failed. Available memory/time remain execution constraints, without creating a speed target. Use a smaller representative proxy for development, preserve original dimensions/weights, and keep full-model acceptance pending until actually executed.

If a required case still differs, use [diagnostics.md](diagnostics.md), refine the local arithmetic/state mapping and rerun it. Do not claim that a seed, deterministic flag, common backend, higher precision or this Markdown procedure alone guarantees equality. Retain the measured artifacts required by [evidence.md](evidence.md).

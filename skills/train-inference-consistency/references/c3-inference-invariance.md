# C3 — Implement and validate inference invariance

**Entry:** C2 proxy and a declared inference matrix. This gate establishes inference properties; C5 separately establishes train/rollout parity.

## Develop the numerical path

1. Follow [correctness-first.md](correctness-first.md). Select or implement target-project providers according to the numerical contract: GEMM/attention reductions, norms, RoPE, activations/residuals, head/logprob, stateful blocks and collectives. Check dispatch over real shapes and prevent unsupported fallback from being reported as strict execution.
2. Keep explicitly required batching, cache, graph and prefill/decode behavior. Choose simpler slow implementations and disable optional accelerations as needed. If caching/Graph is unspecified, record the chosen implementation and tested scope at C0; an uncached implementation may be the final path but establishes no cached-route claim. Serial internal processing must still exercise required external batch cases.
3. Implement capture with logical token/state mappings and original-dtype bytes. Retain actual worker routes and expected rank/shard coverage. Check masks, positions and cache ownership before interpreting numerical differences.

## Execute the matrix

Include repeated identical execution and all required combinations of:

- Request alone versus different batch sizes/neighbours, uneven lengths, padding/packing and request permutations.
- Full versus chunked prefill; equivalent token scoring across prefill/decode, including split points and applicable graph capture modes.
- Mixed prefill/decode scheduling, cache reuse/reorder, prefix-cache hit/miss and equivalent reset state.
- Supported sequence/shape boundaries, partial tiles and relevant topology/communication configurations.
- Model-specific cases: Hybrid chunk/recurrent and convolution/state resets; MoE independent routing, empty/skewed experts, dispatch/combine and capacity behavior.

Declare unavailable or inapplicable cases before acceptance with a scope reason. Never omit required cases after observing failures. Reset equivalent state, independently run every side and repeat, then compare by logical identities. Require finite, nonempty, complete captures and raw-byte equality; retain router/state evidence as applicable.

## Artifacts and exit

Retain provider implementations/guards, regression cases, matrix, independent captures, byte comparison results, coverage receipts and loaded-worker identities. Pass only when all required inference pairs match within the declared scope.

On failure, follow [diagnostics.md](diagnostics.md), preserve the smallest failing fixture and implement a targeted fix. Rerun that case plus affected shape/cache/routing/repeat cases; keep missing coverage blocked. Equality of final generated text alone cannot pass. Continue directly to C5 in consistency-only mode. Continue through C4 only when performance work is requested.

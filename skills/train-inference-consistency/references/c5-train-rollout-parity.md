# C5 — Match both independent training forwards to rollout

**Entry:** C3 path, C2 proxy and a frozen policy version theta0; also require C4 if the performance track is enabled. Execute the real target training and rollout engines, with correct checkpoint conversion and synchronization.

## Integrate the actual branches

1. Trace where the training framework chooses stored rollout logprobs versus recomputation, and where actor/current logprobs enter the loss. Inspect relevant flags and actual worker callsites. A stored value path cannot satisfy independent old-logprob recomputation.
2. Bind rollout, old-logprob and actor to the same synchronized weights and exact tokens, masks, positions, prefix/state and scoring definition. Audit master-to-deployment precision, tied weights, TP/CP/EP partitions and gather/unpermute ordering.
3. Implement compatible forward providers/glue in the training path, preserving a usable autograd graph for actor. A common slow arithmetic implementation shared by all three branches is allowed; independent computations and state are still required. Preserve the target's required train/eval, dropout, autocast, graph and cache semantics. Resolve stochastic differences explicitly; do not silently disable a required behavior.
4. Capture rollout values at generation/scoring time. Independently recompute old logprobs from the fixed token sequence. Independently execute the actual grad-enabled actor forward at theta0 before an update. Record executed callsites, phase/mode and output gradient connectivity.
5. Compare rollout↔old and rollout↔actor separately using original-dtype bytes and complete logical coverage. Include model-specific router/state/dispatch evidence, not just final selected scores. Exercise the production batching/packing/parallel scope, not only batch one.
6. Add target-project tests that catch copied rollout values, detached actor outputs, wrong token/loss alignment, missing workers and stale weights. Preserve both successful receipts and deliberate failure-case results separately from model acceptance captures.

## Artifacts and exit

Retain both independent comparison families, raw captures, weights/fixture identities, actual forward bindings and callsite receipts, gradient connectivity, logical masks/positions and coverage. Both branches must run and match rollout at the same policy version. An equal generated sequence, inference-only surrogate, zero comparisons or one passing branch cannot pass C5.

For MoE, routing replay can diagnose an expert path but cannot establish independent router equality. Keep replay results separate; final C5 requires independently calculated expert choices/weights and correct dispatch/combine mappings.

On failure, identify the failing branch and follow [diagnostics.md](diagnostics.md). Repair weight conversion, score semantics, state, forward math or capture as evidence warrants. Return to C0/C3 where the repair changes identity or inference math; revisit C4 only if performance work is enabled. Leave C5 pending/failed if the actual actor route cannot be executed, even when a reference backend matches.

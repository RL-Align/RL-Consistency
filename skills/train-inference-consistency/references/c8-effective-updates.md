# C8 — Verify two effective updates on the proxy

**Entry:** C7 and the actual target training objective/optimizer. A negative-logprob or other surrogate calibration loop only establishes that calibration scope, not the requested RL training loop.

## Execute the versioned loop

1. At theta0, synchronize rollout/training weights and deployment representation. Independently compare rollout↔old and rollout↔grad-enabled actor with full required routing/state evidence, as in C5.
2. Execute the actual loss, backward and optimizer step. Retain finite loss, gradient participation/finiteness and nonzero gradient/update evidence. Record optimizer state and actual step execution; a printed step counter is not sufficient.
3. Fingerprint complete master parameters before/after and measure a finite nonzero master update norm. Record separately the deployed weights/version and conversion. Low-precision deployment bytes can remain unchanged after a real master update; their unchanged bytes do not establish a failed master update or prove synchronization.
4. Publish/synchronize theta1 through the real target mechanism. Verify observed worker weight versions; invalidate/rebuild caches, recurrence and prepared-weight state that belong to theta0. Independently execute the theta1 rollout/old/actor comparisons.
5. Execute a second effective update with the same evidence, synchronize theta2, and independently repeat its comparisons. Preserve all three versions' records and normal process exit.

Compare only within one policy version. Do not expect theta0 scores to equal theta1 scores after training. In PPO-like inner updates, stored old logprobs intentionally remain tied to the earlier behavior policy; an actor at newer weights need not equal them. The theta1/theta2 parity checks require newly synchronized same-version rollout/reference cases, while the actual training loop preserves its intended old-policy semantics.

## Artifacts and exit

Retain the three dual-path comparisons, actual objective/config/data inputs, complete master/deployment identities, optimizer step/state receipts, finite/nonzero gradient and update evidence, synchronization/cache checks and normal exit. Pass only after two effective master updates and every required same-version comparison.

If lr=0, an overflow/skipped step, all-zero gradients or unchanged master parameters prevent an effective update, record failure and diagnose it. If synchronization is incomplete or a cache uses old weights, correct that before attributing drift to kernels. Return to C5/C6/C7 when the issue concerns forward, backward or optimizer math. A fixed-policy diagnostic cannot substitute for C8.

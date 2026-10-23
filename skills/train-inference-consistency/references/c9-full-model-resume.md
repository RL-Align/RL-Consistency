# C9 — Accept the full model, declared matrix and resume

**Entry:** proxy C5/C7/C8 evidence and the accepted full-model recipe. Verify the relevant earlier results still bind the current implementation. Record the actual recipe decision; do not invent approval.

## Execute full-size acceptance

1. Load the real full checkpoint with all required layers and source dimensions. Verify tensor manifests, converters, trainable/deployment state, objective and actual worker providers. A tiny architecture or reduced proxy cannot satisfy full-model acceptance.
2. At theta0, run both independent old-logprob and grad-enabled actor comparisons against rollout before escalation. Stop and diagnose any mismatch; do not proceed with a known bad initial policy merely to finish a step count.
3. Execute the full two-effective-update loop from C8 using the real training configuration. Preserve per-version weight/sync evidence, routing/state coverage, finite gradients and normal exit.
4. Run the required context, batch/packing, cache/PD, precision and topology matrix. Test cross-topology or cross-hardware equality only when explicitly claimed; two independently passing platforms do not establish equality between them. Include memory/capacity limits and all required ranks/experts.
5. Save a checkpoint after an effective update. Retain model/master/deployed weights, optimizer, scheduler, RNG streams, data cursor and relevant training/runtime state. State whether request caches are restored or deliberately rebuilt under the serving contract.
6. Fork uninterrupted and resumed continuations from the same recorded state and next input. Verify the restored identities, then compare the first restored forward outputs/routing/state and the next actual update/optimizer state against the uninterrupted continuation under the declared resume contract. Exact same-environment resume claims require the stated exact state comparisons; any allowed numerical tolerance must be declared and reported separately.
7. Keep fixed-policy train/rollout comparisons inside the restored run as well. Successful save/load alone proves no resumed numerical behavior.

## Budget checkpoint save and load separately

A steady-state forward/backward/update fit does not prove that checkpoint I/O fits. Inspect the actual optimizer sharding metadata, CPU/GPU staging, gathers and dummy-state initialization. Record allocated/reserved/free device memory before and after loading and on failure. Distinguish framework allocations from other processes or imported allocations; an allocation error alone does not establish fragmentation or an exclusive-device budget.

Prefer an existing memory-efficient CPU/Gloo checkpoint path when the pinned implementation supports it and its process groups are initialized. Preserve optimizer/scheduler/RNG/data state and prove a full restored continuation; a successful small gather or save is only a precheck. Do not discard moments, switch to finetuning, reduce the model or skip the next-update comparison to hide a peak.

Inspect optimizer loaders as well as checkpoint readers. Some versions create dummy moments, invoke a base loader, then reset and allocate moments again for scaled-state handling. In a proven plain-FP32-only configuration this can duplicate the complete GPU state unnecessarily. A narrowly guarded reuse/loading repair must preserve parameter ordering, group metadata and actual optimizer steps, reject unsupported scaled/low-precision/capturable states, pass original-provider load/next-update byte comparisons, and finally pass the full uninterrupted-versus-restored state comparison. Never bypass scaled-state handling for an unverified configuration.

## Artifacts and exit

Retain full-model config/manifests, accepted recipe, initial dual-path entry, complete update loop, predeclared matrix with expected/observed coverage, full worker identities and both uninterrupted/resumed captures plus state receipts. Pass only when all required cases and first-restored forward/update checks meet their explicit criteria.

On a full-model-only failure, retain the case, locate its earliest observed frontier and refine the proxy if it missed the responsible interaction. Repeat affected earlier gates and full acceptance after repair. Memory exhaustion, unavailable topology or missing checkpoint state remains an explicit blocker/failure; a shorter context or fewer layers cannot silently satisfy the original matrix.

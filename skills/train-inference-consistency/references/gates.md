# C0–C10 gates

These IDs belong to RL-Consistency, not RL-Kernel's historical WS1 C numbering.

| Gate | Required evidence | Exit |
|---|---|---|
| C0 | Bound recipe; worker environment/model/fixture identities and loaded artifacts | Identity matches the run |
| C1 | Tuned native benchmark; at least 3 measured steps after warmup | Valid data; freeze after human baseline review |
| C2 | Source/reduced layer specifications; selected layer indices; boundaries | All operator types and adjacent layer patterns retained; no dimension/precision shrink |
| C3 | Batch, padding and permutation trace pairs | Zero differing raw-bit elements after logical alignment |
| C4 | Chunk and prefill/decode trace pairs | Zero differing raw-bit elements |
| C5 | Actual bindings for every operator on both engine roles | Same contracts, observable implementation, no fallback |
| C6 | Rollout/training traces computed independently | Same weight version/input/scoring, zero differing bits |
| C7 | Forward and gradient independent-reference traces | Human-reviewed tolerance passes; no shared tested implementation |
| C8 | At least two real updates and post-sync comparisons | Changed weight hashes, linked versions and parity after each update |
| C9 | Unmodified full model, context/topology coverage, restored states and first step | Full required case matrix; resume forward and first update agree with uninterrupted control |
| C10 | Strict benchmark against C1 | Same work/settings; report overhead and memory; no automatic performance threshold |

Dependencies: C1 and C2 branch from C0; C2–C9 are sequential; C10 joins C1/C9. Precision review of C0 precedes C7. Baseline review of C1 precedes C10. Release review requires all gates.

A failed rerun replaces the latest verdict and invalidates dependent verdicts. Artifact mutation invalidates the affected attempt. Do not erase a failing attempt to retain a prior pass. Lock a new recipe/run when sources or scope change.

# Numerical contract

Specify block geometry, within-block accumulation, cross-block/rank reduction order, accumulator type, rounding points, fusion/FMA rules, transcendental implementation, quantization/scales and logical data ordering. Hardware transport must preserve values and logical identities. A fixed rank tree alone does not guarantee equality when partitions change; preserve canonical arithmetic boundaries as well.

Distinguish repeatability, batch invariance, prefill/decode invariance, train/rollout parity, cross-topology equality, cross-hardware equality and optimizer-trajectory equality. Validate each advertised relation separately. The initial project implements the within-backend workflow.

Compare raw little-endian IEEE/BF16 bytes with dtype and shape checks. Distinguish positive and negative zero. Reject NaN/Inf even when bit patterns agree. Empty comparisons, duplicated token identities and missing tokens fail. Token generation equality, rounded printed values and aggregate tolerances are not bitwise evidence.

Align by logical sample ID, token position and semantic node. Remove padding only by the recorded valid mask. Preserve packed-sequence boundaries, causal positions, model weight version and tokenizer identity. Logprob semantics must match temperature, top-k/top-p support and normalization. Sampling RNG replay and distribution equality are separate from selected-token logprob parity.

Use a same-order reference to check the numerical contract and an independently implemented, higher-precision mathematical reference to check accuracy. State the reference scope. Never claim all parameter gradients are validated when only a subset is captured. Choose and approve tolerances before evaluating results. Keep failing cases.

MoE requires additional router decisions, tie-breaking, dispatch permutation/inverse mapping, expert identities, shared-expert paths and logical return/reduction order. Prefill/decode separation adds KV layout/precision/position and weight-version checks. Add adapters and case matrices before advertising these capabilities.

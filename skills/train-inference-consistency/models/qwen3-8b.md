# Qwen3-8B development guide

Use the common C0–C10 stages for the user's actual Qwen3-8B checkpoint, engines and hardware. This guide supplies Dense-specific work; it does not supply a kernel implementation or establish full-model acceptance. Bind the real config/revision and weight manifests before relying on architecture facts.

## Inspect and map

Inventory embedding/head, decoder layer sequence, Q/K/V and output projections, Q/K normalization, rotary positions, causal attention/KV cache, RMSNorm, gated MLP, residuals and logprob/loss reductions. Confirm actual dimensions, epsilons, activation, positional parameters, tied weights, dtypes and converter transposes from checkpoint and loaded code.

Map every operation to both training and rollout providers and their backward paths. Record training/rollout TP/CP/PP/DP groups and physical sequence layouts. Do not copy a topology or precision contract from another experiment merely because the model name matches.

## Reuse and implementation focus

| Boundary | Development and validation work |
|---|---|
| Projections/GEMM | Batch/shape-dependent algorithm selection, accumulation partitions, prepared weights and casts |
| RMSNorm/QK norm | Epsilon/scale convention, reduction order, accumulation and output rounding |
| RoPE | Logical token positions, rotary dimensions, layout, sine/cosine provider and application/cast order |
| Attention/KV | Mask/scaling/softmax/AV order, cached prefill/decode arithmetic, split-KV and cache state |
| Output/residual/MLP | Output projection, SiLU/gate/up/down composition, fusion and residual rounding |
| Head/logprob | Vocabulary partition/gather, FP32 or declared accumulation, normalization and selected-token alignment |
| Communication | Local partitions, rank reductions, sequence reconstruction and gradient normalization |

Use [correctness-first.md](../references/correctness-first.md) for a common explicit Dense path; per-token projections, ordered norms/attention and serialized reductions may be final implementations. Reuse an existing provider only after checking actual semantics and shapes. The skill has no mandatory external kernel dependency; the agent can reuse compatible available implementations or develop missing ones in the target project.

## Proxy and required cases

Start with a contiguous several-layer prefix if the inspected decoder pattern is uniform. Keep all per-layer dimensions/precision, embedding, final norm/head and selected source weights. Prove coverage in C2; a small hidden-size random model is a separate tooling fixture.

At C3 cover repeat and the declared batch/neighbours, padding/packing/permutation, chunked prefill, cached decode, graph and sequence/topology cases. Explicitly required modes remain mandatory; unspecified accelerations can be disabled and recorded at C0. At C5 retain both independent old-logprob and real actor comparisons; matching inference repeats is insufficient. At C6/C7 cover input/parameter gradients and collective scaling. At C8/C9 verify effective updates, actual weight synchronization, cache/weight-preparation invalidation and restored execution.

Any prior Qwen3-8B result is reusable only with its original model/environment/scope/evidence and a justified compatibility check. This skill project currently records no complete C0–C10 full-model acceptance package. Report guide coverage separately from results of the target run.

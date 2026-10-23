# Numerical contract to implement in the target project

Bind a contract to actual input domain, model/weights, source/build, environment, output semantics and execution paths. Distinguish repetition, batch invariance, prefill/decode invariance, train/rollout parity, cross-topology parity, cross-hardware parity and optimizer-trajectory parity. Test each advertised relation independently.

## Define arithmetic before selecting kernels

| Component | Inspect and specify | Implementation consequence |
|---|---|---|
| GEMM/projections | Operand layout, K partition, accumulator type, tile dependence, rounding and fusion | Fix logical accumulation boundaries/order where required; test batch-dependent dispatch and workspace algorithms |
| Norms/softmax/logsumexp | Reduction axis/order, epsilon placement, scale convention, accumulation and final cast | Match parallel reductions and casts, including the score/logprob path |
| Attention | Q/K scaling, masks, RoPE positions, score/softmax/AV order, split-KV, KV precision | Align cached/decode and full/prefill arithmetic; disabling one path is a scope change, not evidence for it |
| Activation/residual | Exact function, FMA/fusion behavior, gate order, addition and rounding | Match intermediate dtype and fusion boundaries, not only the final dtype |
| Quantization | Scale computation, grouping, rounding, clipping, stochastic state and dequantization | Include scale generation and master/deployment mapping in the contract |
| Collectives | Logical partitions, local accumulation, rank ordering, reduction tree and casts | Fix local arithmetic and global combination; a fixed rank tree alone cannot reconcile changed partitions |
| MoE | Router math, ties, selected weights, dispatch mapping, expert math, drop/capacity policy, combine order | Preserve discrete decisions and logical token/slot identity; unordered accumulation is insufficient |
| Stateful blocks | State definition/precision, update order, chunk boundaries, reset/reorder and weight ownership | Compare equivalent state and invalidate stale caches after weight changes |
| Backward | Differentiated function, saved tensors, casts, gradient accumulation/reduction and any STE | Reuse only semantically compatible gradients; validate independently at C7 |

A deterministic flag or seed alone does not establish these properties. A supported kernel can change algorithm with shape, sequence length, workspace, graph capture or hardware. Inspect dispatch conditions and loaded implementations; pin compatible choices or implement missing paths and guards in the target project. Do not claim an arithmetic guarantee from a configuration flag without exercised evidence.

Transport must preserve logical identities and payload bytes; any in-transit floating-point reduction belongs in the arithmetic contract. For changing topologies, preserve canonical mathematical partitions as well as reduction order, or explicitly restrict the supported topology relation.

## Align what is being compared

Bind sample/request ID, response token position, sequence boundaries, selected token ID, valid/loss mask, positions, scoring semantics, weight version and semantic node. Match tokenizer output and weight conversion, including tied weights, transposes and expert ownership. Do not align by timestamps or incidental memory order.

Declare whether logprobs describe the raw model distribution or a temperature/top-k/top-p transformed distribution. Compare the same definition on both sides, including normalization precision. Greedy token selection does not make its sampling distribution identical to raw-model softmax. Sampling RNG reproducibility is a separate claim.

Compare original-dtype bytes with dtype/shape/layout metadata, including signed zero. Reject NaN/Inf, missing/duplicate logical entries, empty coverage and inconsistent masks. FP64 conversion, decimal printing, a norm of the difference, matching argmax or tolerances cannot establish bitwise equality.

## Preserve accuracy and intended execution

Use a same-order reference to debug implementation of this contract. Separately use an independently implemented higher-precision mathematical reference for forward and gradients; state its limitations. Define tolerances and accepted approximation/STE semantics before examining failures. A repeated error can be deterministic and still wrong.

Keep the user-requested engines, batching/cache modes, topology and objective. Use [correctness-first.md](correctness-first.md) to implement common arithmetic, explicit reductions and sequential evaluation as valid final choices without a performance target. Shared code is allowed; each branch must recompute its own values. Forced routing remains diagnostic and cannot prove independent routing. Choose unspecified optimization modes at C0 and record them; changing an explicitly required engine, device, cache/topology behavior or objective requires a scope decision. Do not transfer results to an unexecuted route.

# Qwen3-30B-A3B MoE development guide

Bind the actual `Qwen3-30B-A3B` revision/config/weights and loaded `qwen3_moe` implementations. Verify expert count, top-k, expert width, normalization, auxiliary objective and quantization; do not substitute a different variant or assume the framework's example defaults match the intended contract. This guide provides development requirements, not full-model acceptance.

## Proxy and common work

For a uniform repeated MoE decoder, start with a contiguous four-layer proxy if it covers the observed layer families and transitions. Preserve all original per-layer dimensions, routers, expert count/top-k, shared experts if present, embedding/head and selected source tensors. For an inspected 128-expert/top-8 checkpoint, keep 128/top-8; reducing experts changes the tested problem. Add coverage for inactive/empty/skewed experts and actual expert partitions.

Reuse compatible Dense components only after checking norm, RoPE, attention, GEMM, MLP, head/logprob and collective contracts. Inspect ordinary versus offset norm scale conventions explicitly; Hybrid and Dense implementations are not interchangeable by name.

## Bind actual module and converter layouts

Inspect the loaded object tree before selecting adapter symbols. A vLLM build may expose a `MoERunner` that owns `routed_experts`, while Megatron may store sequential experts under `local_experts`; neither layout can be inferred from an older fused-MoE class name. Map both layouts to canonical global expert IDs and verify every gate/up/down tensor after conversion and after each actual weight reload. Loader warning lists alone cannot establish whether expert values were loaded; compare the actual parameter bytes.

Derive Q/K/V widths from head count and head dimension separately from hidden size. In the inspected 2048-hidden, 32-query/4-KV-head, head-dimension-128 geometry, Q width is 4096 and each K/V width is 512. A dense-model split copied from another configuration can silently corrupt the mapping.

## Implement and observe MoE semantics

| Boundary | Required work |
|---|---|
| Router | Projection, softmax precision, top-k tie behavior, selected-weight normalization and cast/multiply order |
| Dispatch | Stable logical request/token/layer/slot/expert mapping, packing/permutation and inverse mapping |
| Transport | Actual EP/TP groups, rank-to-expert ownership, payload identity, empty/skewed sends and device-supported collectives |
| Expert computation | Per-expert GEMM/activation/casts, shape-dependent reductions and independent expert microbatches |
| Combine | Logical return ordering, weighted accumulation and fixed ordering where the contract requires it |
| Objective/backward | Router/expert/shared-path gradients, auxiliary losses, active parameter coverage and distributed scaling |

Use [correctness-first.md](../references/correctness-first.md): sequential expert evaluation and ordered combine are valid final choices without a speed target. Choose and implement a deterministic exact-tie rule when required; do not assume native `topk` guarantees it. Record normalized weights before and after applied casts as appropriate. Inspect the actual accumulation implementation: a recorded ordering list does not make an unordered atomic combine invariant.

Record dense/expert TP, EP, DP and PP groups independently. These groups can overlap on the same ranks; do not multiply nominal TP and EP dimensions to infer GPU count. Verify real weight redistribution and expert ownership between training and rollout.

## Routing evidence and failure isolation

At C3/C5/C8/C9 independently capture router logits, selected expert IDs/weights, dispatch/unpermute mapping, expert identity and combine order for every required token/layer/rank. Predeclare prompt and decode coverage, not only scored response tokens. Bind the route to the same execution/weights/input as its output capture.

Prefix-cache hits can omit prompt computation for a later request even when its response logprobs match. Trace the actual cache-construction execution, version, logical aliases and state provenance if those cached prompt routes count toward coverage. Otherwise add independent fresh-prefix evidence, or choose training prefix caching off at C0 when it is unspecified, while retaining every required cache-enabled inference case. Never fill a missing prompt boundary with copied training routes or count an unobserved boundary as equal.

For backward coverage, bind each expert to the actual selected token rows and its upstream output cotangent. Distinguish an unvisited expert, a visited expert with zero upstream gradient, and an active expert whose required gradient is missing. Preserve and account for zero/absent contributions; a blanket requirement that every original expert have a nonzero gradient in every microbatch is invalid for conditional MoE execution.

If the contract forbids token dropping, verify actual counts/capacity behavior rather than trusting a policy name. If dropping is intentionally part of the model, define and compare its discrete semantics explicitly. Retain both independent old-logprob and grad-enabled actor comparisons. Replaying rollout routes into training is useful diagnosis but cannot pass independent routing.

On mismatch, inspect router inputs→logits→softmax→IDs/weights, dispatch, expert outputs and combine. Equal final logprobs do not excuse differing required routes. C7 needs independent router/expert/combine math and gradient tests away from top-k discontinuities; test exact ties separately. Record the actual auxiliary-loss objective rather than silently zeroing it to obtain parity.

After effective updates, verify synchronized router/expert weights, same-version independent routes and output comparisons, then full-model context/topology and uninterrupted/resumed behavior. Profile dispatch/all-to-all, grouped GEMM and combine against the frozen native baseline only if performance work is explicitly requested. Missing device providers, capture or full-model evidence remains an explicit blocker.

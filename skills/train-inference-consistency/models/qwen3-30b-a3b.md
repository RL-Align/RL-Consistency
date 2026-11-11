# Qwen3-30B-A3B MoE development guide

Bind the actual `Qwen3-30B-A3B` revision/config/weights and loaded `qwen3_moe` implementations. Verify expert count, top-k, expert width, normalization, auxiliary objective and quantization; do not substitute a different variant or assume the framework's example defaults match the intended contract. The requirements below guide new implementations. A completed full-model run is recorded in the measured profile below.

## Current validation — 2026-11-11

This table is the current task result; dated historical sections below retain their original scope. See the [current evidence](evidence/rocm-20261111.md) and [exact archive/replay instructions](evidence/rocm-20261111-reproduction.md). No RL-Kernel dependency. Performance: not evaluated.

Measured hardware: AMD Instinct MI300X VF (gfx942), physical Devices 4–7. Measured configuration: original revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`,48 layers/30,532,122,624 parameters,128 experts/top8. BF16 model/activations/KV, FP32 gradients/masters/moments and GPU FP64 Adam intermediates. Native training PP3 on Devices4/5/6, independent rollout TP1 on7; TP/CP/DP/EP1. Main chain microbatch1/padding64, globalbatch2,20-token prompts and16 responses. The microbatch2/padding128 layout passes against microbatch1/padding64. Full inference covers8/20/32-token prompts,16 responses, cache off/on and chunks64/16; cache-on uses explicit2GiB native KV capacity.

| Gate | Status | Actual scope/evidence |
|---|---|---|
| C0 | PASS | Original48 layers/18,867 logical tensors; actual PP3 and independent rollout provider/device inventories |
| C1 | NOT REQUIRED | Performance: not evaluated |
| C2 | PASS | Four original-width layers,128 experts/top8; proxy update/route/backward and cache-transition coverage |
| C3 | PASS | Proxy and original48-layer four-configuration matrices pass482 pairs/1,952 score tokens each; full independent MoE boundaries pass |
| C4 | NOT REQUIRED | Performance: not evaluated |
| C5 | PASS | Full model: all three independent pairs96 valid tokens/pair;181,440 MoE field comparisons |
| C6 | PASS | Full model:39,518 native gradient hooks,288 actor layer forwards and exact active-expert/cotangent coverage |
| C7 | PASS | Independent router/expert/combine/gradient references; every full optimizer element checked at fixed accuracy criteria |
| C8 | PASS | Three full-model updates; actual BF16 deployed weights change at both required boundaries |
| C9 | PASS | Complete fork and fresh continuation pass: all scores,13,198 gradient tensors,30.532B optimizer elements,18,867 model/deployed tensors and RNG/scheduler/data equal; layout passes64 score tokens and6,720 MoE rows; complete full matrix passes |
| C10 | PASS | Base archive and exact native-KV supplement delivered; reconstructed full restore/layout/matrix and unsupported-config guards pass; final English replay instructions verified |

The declared reconstruction scope uses the pinned installed Python/native ABI and an isolated tree reconstructed from the delivered archive. Other hardware/topologies, arbitrary contexts, quantization, graph/v2-runner modes and performance are not covered. Consistency-only acceptance is complete in the declared scope. C1/C4 are not required; performance is not evaluated. Release proceeds through a PR subject to repository checks; the final document/PR/merge binding is retained in the external release-event.json.
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

## Published evidence receipts

See the [measured environment and original receipt table](evidence/rocm-20261023.md#qwen3-30b-a3b) for the dated acceptance status, raw comparison excerpts, exact source bindings, artifact paths and SHA-256 digests. Complete acceptance and in-progress measurements are explicitly distinguished.

## Measured ROCm run — 2026-10-23 to 2026-10-24

The full 48-layer Qwen3-30B-A3B implementation completed bitwise training–inference, effective-update and checkpoint-resume verification in the measured MI300X ROCm environment. The [published environment and evidence tables](evidence/rocm-20261023.md#qwen3-30b-a3b) contain original receipts and artifact SHA-256 digests.

| Measured item | Result |
|---|---|
| Post-training and rollout | Native vime/Megatron GRPO/PPO and independent vLLM rollout; exact source bases and task patches are recorded in the evidence page |
| Model | Official revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`; 48 original layers, 128 experts/top-8, 30,532,122,624 parameters; all 18,867 source tensors verified |
| Environment | Two AMD MI300X VF/gfx942 devices; PyTorch2.12.0/ROCm7.14; TP/CP/PP/DP/EP1; eager legacy runner |
| Precision | BF16 model/activations/paged KV/main-gradient buffers; CPU FP32 AdamW masters/moments; common arithmetic, independent routes and ordered combine |
| Full inference matrix | Batch1/2, 8/32-token prompts plus16 responses; repeat/single/permutation, fresh teacher prefixes, cache off/on and chunks64/16 pass |
| Training–inference equality | Each of theta0/theta1/theta2: rollout-old and rollout-grad-enabled-actor each pass32 positions and3,360 logical MoE layer-rows x9 fields |
| Actual updates | Every element of all three original updates and restored next update passes the independent FP64 Adam reference; real backward/accumulation/norm/clipping and loss references pass |
| Checkpoint and resume | Native398.245GiB checkpoint; uninterrupted `vime-chain.0j028Z` vs restored `vime-chain.8cpTEe` match raw scores, complete optimizer/deployment states, real gradients, RNG/scheduler and next actual update |
| Final runtime result | Both normal exits0; resume and MoE resume `all_equal=true`; final runtime flag0 at2026-10-24 00:09:32 Asia/Singapore |

Scope is limited to the recorded modified providers and measured matrix. Training uses microbatch1/globalbatch2, two20-token chat prompts plus16 responses and THD padding128, with prefix caching off. Independent accuracy uses frozen tolerances separately from original-byte equality. Other hardware, frameworks, arbitrary contexts, multi-rank execution, quantization, graphs and the v2 runner remain unmeasured. Performance was not evaluated. Final release-bundle bookkeeping is separate from the completed runtime receipts.

## Current execution — 2026-11-06

The November task uses physical Devices 4–7 only and does not inherit acceptance from the historical run. See the [current gate table](../../../STATUS.md), [execution receipts](evidence/rocm-20261106-resumed.md), [storage finding](evidence/rocm-20261106-storage.md), and [reconstruction evidence](evidence/rocm-20261106-reconstruction.md). Performance: not evaluated.

| Current measured item | Result |
|---|---|
| Environment | Same pinned ROCm/PyTorch/vime/Megatron/vLLM environment as the current dense run; task-private ROCr and inherited Landlock isolation |
| Model / precision | Original48-layer revision bound; measured proxy retains4 original-width layers,128 experts/top8; BF16 model/activations/KV and FP32 main gradients/masters/moments |
| Parallelism | Proxy training PP3 on physical4/5/6, TP/CP/DP/EP1; independent TP1 rollout on7; full-model PP3 remains unverified |
| C0/C3/C9 | NOT RUN for complete required full-model coverage |
| C2/C5/C6/C7/C8/C10 | BLOCKED; partial receipts below do not accept these gates; C1/C4 not-required |
| Independent byte equality | Proxy initial version only:32 valid score tokens per pair and5,040 router/dispatch/expert/combine field comparisons pass |
| Backward / effective update |1,402 actual gradient-hook payloads and667 participating-expert records audited;337 completed parameter references pass, but no all-rank optimizer completion or valid updated checkpoint |
| Sync / full checkpoint restore | Updated-version synchronization loop and complete48-layer fresh-process restoration NOT RUN |
| Documentation reconstruction | Shared source reconstruction passes; no accepted full-model MoE entrypoint reproduction |
| Blocker / scope | Confirmed UID1000 block quota on `/dev/vdc1`; failed attempt and partial outputs retained. No full-model bitwise acceptance or performance claim |

Resume the incomplete four-layer native loop only after sufficient quota headroom is confirmed. Then complete the required matrix, full-model updates/synchronization, checkpoint restoration and documentation reproduction. The failed partial optimizer state is not a valid resume checkpoint.

## Continuation — 2026-11-11

See the [current continuation evidence](evidence/rocm-20261111.md) for the checked dense-result reuse, new MoE inference measurements, original-byte storage representation and remaining acceptance. Performance: not evaluated.

# Qwen3-8B development guide

Use the common C0–C10 stages for the user's actual Qwen3-8B checkpoint, engines and hardware. This guide supplies Dense-specific work; it does not supply a kernel implementation or establish full-model acceptance. Bind the real config/revision and weight manifests before relying on architecture facts.

## Current validation — 2026-11-11

This table is the current task result; dated historical sections below retain their original scope. See the [current evidence](evidence/rocm-20261111.md) and [exact archive/replay instructions](evidence/rocm-20261111-reproduction.md). No RL-Kernel dependency. Performance: not evaluated.

Measured hardware: AMD Instinct MI300X VF (gfx942), physical Devices 4–7. Measured configuration: original revision `b968826d9c46dd6066d109eabc6255188de91218`,36 layers/8,190,735,360 parameters, BF16 model/activations/KV and FP32 gradients/masters/moments, native FP32 Adam. Training/rollout use TP/CP/PP/DP/EP1 on Devices4/5; the layout case uses6/7. Microbatch2/padding128 and microbatch1/padding64, globalbatch2; inference prompts8/20/32 with16 responses, cache off/on and chunks64/16.

| Gate | Status | Actual scope/evidence |
|---|---|---|
| C0 | PASS | Exact original36-layer identities, worker/operator inventory and allowed GPUs |
| C1 | NOT REQUIRED | Performance: not evaluated |
| C2 | PASS | Original-width source-preserving proxy and actual native chain |
| C3 | PASS | Full cache/chunk/batch/packing/teacher-prefix matrix;482 pairs/1,952 tokens |
| C4 | NOT REQUIRED | Performance: not evaluated |
| C5 | PASS | All three independently computed score pairs:96 valid tokens/pair across theta0/1/2 |
| C6 | PASS | 873 actual native gradient-hook records; finite gradients and padding coverage |
| C7 | PASS | Independent common/loss/gradient references and complete restored optimizer-step audit |
| C8 | PASS | Three actual updates; real model changes and complete deployment identity checks |
| C9 | PASS | Full checkpoint fresh restore: first forward, gradients, effective next update and complete related state match |
| C10 | PASS | Actual delivered archive reconstruction, full8B entrypoint32 raw tokens and unsupported-config rejection; English replay instructions |

The declared reconstruction scope uses the pinned installed Python/native ABI and an isolated tree reconstructed from the delivered archive. Other hardware/topologies, arbitrary contexts, quantization, graph/v2-runner modes and performance are not covered. Consistency-only acceptance is complete in the declared scope. C1/C4 are not required; performance is not evaluated. Release proceeds through a PR subject to repository checks; the final document/PR/merge binding is retained in the external release-event.json.
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

Any prior Qwen3-8B result is reusable only with its original model/environment/scope/evidence and a justified compatibility check. The measured run below records one environment-bound full-model acceptance. Report guide coverage separately from the results of any other target run.

## Measured ROCm run — 2026-10-23

C0–C10 consistency-only acceptance completed for the following actual target implementation; C1/C4 not-required and performance not evaluated. This skill remains Markdown-only; the implementation, source patch archives and raw evidence are retained in the user's target workspace, not bundled as a runtime here.

| Axis | Measured scope |
|---|---|
| Model identity | Original 36-layer checkpoint, hidden4096/FFN12288, Q32/KV8/head128, vocabulary151936, BF16 untied head, RMS1e-6/RoPE1e6; all399logical tensors roundtripped byte-exactly;8190735360parameters |
| Providers | Actual vime base d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36 plus retained implementation patches; Megatron1dcf0dafa884ad52ffb243625717a3471643e087 plus selected AMD patches; original image vLLM11d76fc7259b3c1d70e2043b7513226aec9a665a/compiled extensions |
| Environment | Two MI300X VF gfx942 devices, one training and one rollout; torch2.12.0+rocm7.14.0a20260608, HIP7.14.60850, TE2.14.0.dev0+e6ede467, RCCL2.28.9, Ray2.56.1; fresh worker source/device/extension identities retained |
| Arithmetic | BF16 deployment/activations, orderedFP32projection backward and explicit FP32attention/norm/RoPE/SwiGLU; FP64log-softmax/entropy/backward with finalFP32rows; FP32Adam masters/grad/moments |
| Execution | TP/CP/PP/DP1, eager legacy runner; external inference batches1/2, exact8/32token prompts with16generated tokens, repeat/single/neighbour/permutation, ragged/chunk64/16 scheduling, cached decode versus fresh prefix and cache off/on with observed16token hits; configured max64 and measured logical length up to48 |
| Training | Actual GRPO/PPO with independently frozen old scores, existing math correctness minus0.001×response character length, entropy0.01/KL0; globalbatch2/microbatch1,20token chat prompt+16response, physicalTHDpadding128; Adam lr1e-6/betas0.9,0.98/eps1e-8/weightdecay0 |

Evidence receipts under task artifact group `rocm-qwen3-20261023`:

- Conversion `convert.rSi0ai/full-roundtrip-identity.json`:399/399original logical tensor bytes exact. Representative4-layer proxy keeps47source tensors and original layer dimensions; revised proxy `vime-chain.ecTRwH` passes its three-version trajectory.
- Complete-model inference `inference-reproduction.M1KUrd` and `inference-reproduction.SYG81l`: all declared matrix/teacher comparisons and cross-cache raw comparisons pass.
- Full uninterrupted `vime-chain.EBCzqV` exits0: independently computed rollout↔old and rollout↔grad actor each32/32raw positions at theta0/theta1/theta2;291packed parameters each have2finite/nonzero actual backward contributions. Every8190735360master element passes independentFP64Adam checks per update. Changed master counts6941811945,6947108023,6944154329; actual advantage counts32/0/32 are explicit, with real entropy gradients/effective updates also in the equal-reward group.
- Full restored `vime-chain.iodtVZ` exits0 from a native106.8GiB checkpoint after the second update. `resume-comparison.json` is exact for all3roles32positions,582backward contributions, all291complete master/gradient/moment before/after fingerprints and next actual update, deployed parameters, Python/NumPy/Torch/GPU/TP RNG and scheduler. Request caches are rebuilt.
- IndependentFP64operator, real loss/logits-gradient and full-element optimizer references pass their predeclared thresholds. The full-model concentrated-entropy failure and corrected same-input replay are retained; numerical equality alone was not accepted as accuracy.

The target delivery contains `release-qwen3-8b.md`, `README.md`, executable launch/comparison/reference scripts, source base/patch/file-hash archives and all failed attempts. C9 required the documented nativeCPU/Gloo save path and a guarded plain-FP32 optimizer-load repair; see [checkpoint memory and restore requirements](../references/c9-full-model-resume.md#budget-checkpoint-save-and-load-separately). Their controlled prechecks were followed by the actual full restored comparison.

This result does not cover other checkpoints, arbitrary contexts, other hardware/topologies, quantized cache, dropout, graphs/v2runner or MoE. It does not establish reward improvement or performance. Reuse requires the exact modified providers/model identities and a scope compatibility check; the model name alone is insufficient.

## Published evidence receipts

See the [measured environment and original receipt table](evidence/rocm-20261023.md#qwen3-8b) for the dated acceptance status, raw comparison excerpts, exact source bindings, artifact paths and SHA-256 digests. Complete acceptance and in-progress measurements are explicitly distinguished.

## Current execution — 2026-11-06

The November task uses physical Devices 4–7 only and does not inherit acceptance from the historical run. See the [current gate table](../../../STATUS.md), [execution receipts](evidence/rocm-20261106-resumed.md), [storage finding](evidence/rocm-20261106-storage.md), and [reconstruction evidence](evidence/rocm-20261106-reconstruction.md). Performance: not evaluated.

| Current measured item | Result |
|---|---|
| Environment | MI300X VF/gfx942; torch2.12.0+rocm7.14.0a20260608, HIP7.14.60850; vLLM98dff2a8; pinned vime/Megatron plus archived patches |
| Model / precision | Original36 layers /8,190,735,360 parameters; BF16 model/activations/KV, FP32 main gradients/masters/moments; FP64 score reduction with final FP32 rows |
| Parallelism | Training TP/PP/CP/DP1 and independent TP1 rollout; physical4/5, with controlled layout test on6/7 |
| C0/C2/C3/C5/C6/C7/C8/C9 | PASS in the declared measured scope; C1/C4 not-required |
| Independent byte equality | All3 score pairs at theta0/1/2,96 valid tokens per pair; full inference matrix482 pairs /1,952 score tokens |
| Effective update / sync | Three actual native updates; full-element independent optimizer references;399 original logical tensors audited at live synchronization; cache invalidation observed |
| Full checkpoint restore | PASS in a fresh process: later inputs/scores,291 gradients, next effective update,1,421 checkpoint chunks and RNG/scheduler/data state match the uninterrupted run |
| Documentation reconstruction / C10 | Source reconstruction and actual reconstructed proxy entrypoint PASS; final portable delivery remains BLOCKED |
| Not covered | Other checkpoints, arbitrary contexts, other topologies/hardware, graphs/v2 runner, quantization and performance |

Current numerical acceptance is supported by November receipts. The two-model release is not complete: MoE gates and final documentation/implementation delivery remain outstanding. Full dense restore is complete and is not a pending experiment.

## Continuation — 2026-11-11

See the [current continuation evidence](evidence/rocm-20261111.md) for the checked dense-result reuse, new MoE inference measurements, original-byte storage representation and remaining acceptance. Performance: not evaluated.

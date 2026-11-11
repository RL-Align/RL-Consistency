# Current acceptance — 2026-11-11

Both Qwen3-8B and Qwen3-30B-A3B pass all required consistency-only gates in the documented MI300X VF/ROCm scope. C1/C4 are not required; performance is not evaluated. Complete independent scores, backward/math, effective updates, real synchronization, fresh complete-state restoration, full inference matrices, layout cases and documented source reconstruction pass. The full MoE matrix closes with `matrix-byte-comparison.lr8tbbl2`, `moe-matrix-boundaries._ne0m13j` and `matrix-reconstructed-entrypoint-audit.cugyb0jx`. All task GPU processes have exited. The final document commit and actual PR/merge result are recorded in `/workspace/validation/continue-20261111-831k95g3/release-event.json`. Historical entries below preserve the state at each timestamp.

# Validation continuation — 2026-11-11

Updated: 2026-11-11 14:41 Asia/Singapore. User renewed the request to obtain both models bitwise consistency on refactor/split-consistency-skills. Physical Devices4–7 only, GPU runs serial. Performance: not evaluated.

The existing 8B complete-model numerical/update/resume receipts remain reusable. MoE full acceptance remains pending. A scoped640 GiB proxy-chain reservation returned EDQUOT and was immediately removed; no historical evidence was touched. Continue independent MoE inference cases rather than waiting for an external quota report. New lossless capture storage verifies fresh bytes completely before deduplicating equal payloads; no computation or required raw tensor is omitted. The prior pending cache-provenance observer is now applied for actual GPU verification.

Current root: `/workspace/validation/continue-20261111-831k95g3`. Prior gate table and evidence are preserved in `previous-STATUS.md`. Next: actual four-layer MoE matrix, original-byte and route comparisons, then remaining feasible full-model cases. Complete optimizer and checkpoint state remain mandatory; storage errors cannot be marked as passing.

## First continuation update

- Dense reuse audit: named numerical/serving/isolation sources, all four repository bases/patches and complete package freeze unchanged.
- MoE proxy inference: all four cache/chunk configurations exit0;482 score pairs/1,952 valid token bytes equal. Independent MoE boundary audit passes39,704 layer-row comparisons,6 fields each,6,812 cross-prefill/decode rows, with dispatch/inverse/coverage verified. Reports: `matrix-byte-comparison.w9ug_62c` and `moe-matrix-boundaries.d501vyn2`.
- Full48-layer matrix running on Device7 in `moe-full-rollout-matrix.h8LpVM6C`;18,867 original logical tensor identities checked. First configuration currently105/112 cases; not yet accepted.
- Independent CPU FP64 checks of the actual first-token capture across all48 layers pass528 checks and exact route choices: `actual-moe-reference.nkyykrit`. This is forward accuracy only.
- Lossless per-row compressed captures preserve original bytes while reducing repeated prefill storage. Old captures and failures remain. The native GPU sandbox is unchanged.
- A bounded-memory native FusedAdam observer candidate is prepared, not yet GPU-tested or installed. It retains full native optimizer state and all-element checks, replacing repeated multi-terabyte temporary FP64 reference files with streamed reference digests/metrics and failure payloads. Native-vs-partitioned byte comparisons are required before use.

Full MoE updated-version three-path, complete backward/update/restore and final C10 remain pending. No merge. Do not ask for another quota report as a prerequisite; continue executable checks while preserving actual storage failures.

## Second continuation update

- Full MoE inference cache-off/chunk64 completed112 cases and122 requests with exit0. Remaining three configurations are pending.
- Actual task-local reservation of427,449,716,736 bytes (one complete30B checkpoint budget,14 bytes per parameter) passed with allocation and fsync, then was removed.640GiB did not fit. Use one complete fork checkpoint and complete subsequent state hashes; no optimizer state is omitted.
- Bounded native optimizer fixture passed three real GPU updates with byte-identical native grouped/partitioned outputs and rejection of an injected wrong update.
- Actual proxy chain `moe-proxy-bounded.IjpxJPQ3` failed the independent Adam moment check on update2:128 elements, scaled error2.0778811318960896e-6 exceeds the unchanged2e-6 limit. First update and subsequent independent score paths executed; the complete chain is FAIL.
- Diagnostic replay `moe-proxy-math-replay.lpM6g8xF` captures complete original inputs/outputs of this failure. A separate GPU FP64-intermediate Adam candidate preserves FP32 persistent states and native group-step lifecycle; it is not yet validated or installed. Native failure remains recorded.
- Dense8B numerical/update/resume evidence remains reusable. Full MoE C7/C8/C9 and final C10 are pending. No merge. Performance: not evaluated.

## Repaired proxy and full-model launch

Proxy `moe-proxy-bounded.DLxZDQRP` exits0 with three genuine updates. Every update audits all3,114,814,464 original proxy parameters; changed FP32 master elements are2,116,482,062 /2,435,270,911 /2,595,691,012. All mathematical checks and complete state continuity pass. Three independent score pairs match96 valid tokens per pair across theta0/1/2 (`three-path-byte-comparison.jhfep_m7`). `moe-three-path-boundaries.v5f47pyy` passes15,120 independent field comparisons. `moe-compact-gradient-comparison.mb2t9edw` verifies4,406 native gradient hooks,24 actor layer forwards and2,101 participating expert records. `three-path-negative.v9tjahe3` rejects an actual one-bit score mutation with its integrity hash updated.

Actual retained theta0 expert gradients also pass independent CPU FP64 matrix calculus for one deterministically selected participating expert per proxy layer (`actual-moe-backward-reference.x7evh1o2`),8 complete-tensor checks at the predeclared0.02 BF16 accuracy tolerance. This does not accept the failed enclosing historical optimizer run.

The optional class-level post-train observer did not emit receipts under Ray. It has been replaced by an explicit MoE-only call at the native train_actor final boundary; numerical providers and algorithms are unchanged. Full runs must contain the actual complete live model/RNG/scheduler receipts. Native optimizer group metadata and permitted state fields are additionally asserted and retained.

Companion `rc-rocm-20261111.1jfv1_zt.tar.gz` has231 entries,97,785,369 bytes, SHA2560078bbda9a9766c8d27d8f3d8aa6a61d8f994a5ffd14d2b64d6e2e442532dabb. Restored source tree `/workspace/consistency-delivery/rebuild-20261111-1jfv1_zt/workspace` matches9,821 original source files plus every acceptance/tooling Python file. This is source/explicit ABI artifact reconstruction, not a complete native dependency recompile.

Full48-layer continuous run `moe-full-bounded.bMdR38Dh` is now executing from that reconstructed source tree on physical Devices4–7. All full MoE gates remain pending until complete receipts are checked; one complete fork checkpoint at rollout1 and fresh-process continuation remain mandatory. No merge. Performance: not evaluated.

## Full-model first-update result and retained cache-provenance failure

`moe-full-bounded.bMdR38Dh` exits1 before its second optimizer update. Initial32-token three-path scores and60,480 MoE field comparisons pass. First actual optimizer update checks all30,532,122,624 parameters at fixed thresholds, with14,645,893,470 changed masters. All three explicit post-train model/RNG/scheduler receipts were emitted. Updated-version32-token three-path scores also match. The whole run is FAIL, not accepted.

`cache-order-diagnostic.json` identifies768 aliases at version2 whose true native constructor is another request in the SAME forward/capture. Both have the identical capture timestamp; the old strict earlier-capture assertion rejects this actual in-batch prefix-sharing case. No cross-version source is substituted and no timestamp is edited. Repair attempt1 adds direct original-byte verification of native paged K/V against current real K/V BEFORE attention, followed by a completed-consumption event. Same-capture provenance requires these raw witnesses, exact slot/prefix/PID identity and ordering; other captures still require strictly earlier time. `test-cold-shared-prefix.sh` is exercising actual cold identical requests on Device7 and negative controls. Full-model rerun is required after verification.

The previous full attempt, its exact sources, first-update receipts and failed cache trace are retained. GPU processes from it exited before this fixture. No checkpoint was saved at the failed pre-fork boundary. No merge. Budget deadline16:33:11SG; reserve final delivery from15:53:11SG.

Cache repair scoped verification: `cold-shared-prefix.YLaagXPH` exits0 with64 actual same-forward aliases,32 exact score tokens and five rejected invalid-witness controls. `cache-provenance-fixture.8vxa2ogs` preserves630 positive field comparisons and rejection of wrong version/process/future constructor/causal input. `moe-common-dimensions.U7SSEvH4` passes24 original MoE dimension references. Repaired full replay `moe-full-bounded.afO1DMSj` is running from restored source `/workspace/consistency-delivery/rebuild-20261111-160aualb/workspace`; artifact SHA25661268ca8fe59e87b8bafdf3dce38320d77e9727c6ec9084292072ec18285d3a9. Earlier source/report artifacts and failed run remain immutable.

## Portable reconstruction and current full replay

The repaired full run `moe-full-bounded.afO1DMSj` remains active from the restored160aualb tree. Actual first-step all-element optimizer reference checks are progressing without an observed failed parameter; no complete-run acceptance is inferred. Historical same-forward capture without native K/V witnesses is still rejected by the repaired comparator (`historical-cache-failure-rejected.json`).

Portable native-chain entrypoints now resolve drivers, observers, model providers, fixtures and provenance capture from their delivered location. Model and evidence roots are explicit environment variables. Artifact `rc-rocm-20261111.8dc__03q.tar.gz` adds a portable full inference matrix entrypoint with an actual worker-provider/unsupported-configuration check; its SHA256 is `a91bd0d1ea4b9176cd49c9a106d86318dd50048e0444d4853e9ed7e01852665d`. Model arithmetic sources are unchanged from the running160aualb tree. Source reconstruction is being verified; execution of these new entrypoints remains pending. Final C9/C10 and merge remain pending. Performance: not evaluated.

## Full checkpoint saved; continuous next step active

Full run `moe-full-bounded.afO1DMSj` has completed two real updates. Each checks all30,532,122,624 parameters independently; changed master elements total14,645,893,470 and16,077,080,265. The complete native iteration1 fork was saved successfully with427,577,923,699 file bytes across9 files, including native metadata. All3 rank writers completed and real rollout/data progress was saved. CPU inspection is reading all69,193 stored chunks; no complete-inspection acceptance is inferred until its normal exit/report. Third continuous forward/backward/update is active for the later restored comparison. The repaired same-forward cache provenance passes the previously failing second-update entry. Full three-version acceptance awaits normal exit and independent audit.

The latest runtime companion is `rc-rocm-20261111.rgogrmsc.tar.gz`,97,884,706 bytes,285 manifest entries,SHA2560530f5c0ab26e85edc8c747abac7f7abf43e838ec3c182181582bb6a707ff32b. Its actual archive has been extracted and reconstructed at `/workspace/consistency-delivery/archive-rebuild-20261111-rgogrmsc/workspace`; all9,821 project source files and acceptance/tooling sources match. Relative to the running160aualb source, production checkpoint save/load bodies are unchanged; an explicit opt-in read-only memory observer wraps them for the upcoming restore, recording allocator/host state without tensor/RNG mutation or cache release. `checkpoint-memory-source-compatibility.json` records this exact conditional addition.

Portable full native chains, matrix/unsupported-config probes, dense conversion and comparison entrypoints are delivered. A new English reproduction page records exact archive/hash, source/dependency/model bindings, commands, physical GPU identities and save/restore budgets. Local Markdown-only/English/link/anchor/gate/template checks pass46 files,171 relative links and18 heading anchors. The actual portable resumed/layout/matrix/entrypoint runs and final gate ledger remain pending. No merge. Performance: not evaluated.

## Fresh restore failure retained; targeted repair1

`moe-resume-bounded.nYYAjIJK` exits1. All3 ranks loaded the complete checkpoint; native GLU merging caught GPU allocation failures and used its existing CPU concatenation fallback. The first restored requests/seeds, old/actor original bytes, and all18,867 restored/deployed logical weights match the continuous run (`resume-initial-diagnostic.json`). This does NOT accept C9. The next optimizer step failed before updating parameters in native `get_grad_norm_fp32` -> RCCL all_reduce: `Failed to CUDA calloc33554432 bytes`. Full traceback and memory receipts are retained.

Repair attempt1 adds an opt-in release of unused task-process PyTorch allocator cache immediately before the unchanged native optimizer.step. It synchronizes owned work, asserts live allocated bytes unchanged, and records allocated/reserved/free memory before and after. No model/gradient/master/moment/RNG state is dropped. New source artifact and fresh-process restore will verify the complete same-next-update comparison; the failed run remains FAIL. Full layout and remaining full inference matrix are still pending. No merge. Performance: not evaluated.


## 2026-11-11 15:09 Asia/Singapore — restored update advancing

Repair attempt1 `moe-resume-bounded.7MUxjxN5` has loaded the complete native fork and executed the next independent rollout/old/actor and backward. All3 rank cache-release receipts preserve live allocated bytes and release7.502 /2.924 /9.223GiB of unused task-process reserves. Native optimizer execution has crossed the retained32MiB RCCL allocation failure; one complete rank mathematical audit has finished and the other two are nearing completion. This is progress, not C9 acceptance: complete exit, continuous/restored state comparison, full layout and remaining full inference matrix remain mandatory. Actual task GPU handles are only physical Devices4–7.

Current source archive is `rc-rocm-20261111.1pwo7oqk.tar.gz`, SHA256 `d941e3bde7c7e3995ef277a79d67605a207aeffd15e92e9a0673a1151067f0ad`; actual isolated source reconstruction verified. The dense full entrypoint used the preceding archive with exact conditional-source compatibility recorded. Current English Markdown/link/anchor checks pass46 files,179 links and18 anchors. The historical quota report is explicitly separated from the now completed427.58GB checkpoint save/readback. No commit/push/PR/merge yet. Performance: not evaluated.


## 2026-11-11 15:39 Asia/Singapore — full restore and layout pass

The full MoE restored continuation `moe-resume-bounded.7MUxjxN5` exits0 and passes complete continuous/restored equality (`moe-bounded-resume-comparison.vpsl4z8m`), including every optimizer element,13,198 gradient tensors,18,867 model/deployed tensors and RNG/scheduler/data. Full layout `moe-layout-bounded.vXU2I1d6` exits0: microbatch1/pad64 versus2/pad128 gives64 original score tokens and6,720 MoE rows byte-identical (`moe-layout-comparison.hc5x1m4x`); its complete independent math and actual reconstructed-provider audits pass. No old task GPU handles remain before the next launch.

The only remaining GPU work is full inference matrix `moe-full-rollout-matrix.OIhGHJWt`, launched15:39SG, physical Device7, serial. It explicitly reuses only the completed cache0/chunk64 configuration and runs the remaining3 configurations with112 cases/122 requests each, including actual reconstructed-provider and unsupported-config guards. C9/C10 remain BLOCKED until complete matrix comparisons and final delivery checks pass. Final evidence/source packaging and English documentation proceed concurrently. No commit/push/PR/merge yet. Performance: not evaluated.


## 2026-11-11 16:09 Asia/Singapore — final matrix memory budget repair

Full matrix `moe-full-rollout-matrix.OIhGHJWt` completed cache0/chunk16 with112 cases/122 requests and exit0, preserving the earlier completed cache0/chunk64. Its cache1/chunk64 engine then failed before scored cases: native vLLM calculated available KV memory as-99.82GiB and raised `No available memory for the cache blocks`. Full traceback and exit1 remain intact. Read-only post-failure checks find no owned live GPU processes and only299,905,024 bytes used on physical Device7. The log lacks complete profiler snapshots, so a specific profiler accounting bug or past transient external allocation is not asserted.

Repair attempt1 uses native `kv_cache_memory_bytes=2147483648` with unchanged model, precision, cache/chunk/batch/context settings and112-case/122-request workload. All122 requests at64 tokens each conservatively need767,557,632 KV bytes;2GiB supports21,840 block-aligned tokens. New continuation `moe-full-rollout-matrix.p5KC7rNY` reuses exactly the two completed cache-off configurations and runs both remaining cache-on configurations on Device7. Original runtime providers are unchanged; the two explicit supplemental launcher/driver files and source-diff/budget evidence will accompany the final evidence bundle. C9/C10 remain BLOCKED until all four configurations, raw comparisons, independent MoE fields and final reproduction checks pass. No commit/push/PR/merge. Performance: not evaluated.

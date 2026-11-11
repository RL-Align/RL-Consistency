# Resumed ROCm validation — 2026-11-06

Status: **IN PROGRESS**, not full-model acceptance. This continuation resolves the strict device-isolation blocker in the [earlier attempt](rocm-20261106.md), whose failures and artifacts remain retained. Neither model has passed every required gate; no commit, push, PR or merge is supported by the partial results below. C1/C4: not-required. Performance: not evaluated.

## Task-private runtime repair

The installed SDK manifest pins [ROCm systems d34cbb640961dcd0c0aa6c1c3c1cfa5d4becfa31](https://github.com/ROCm/rocm-systems/tree/d34cbb640961dcd0c0aa6c1c3c1cfa5d4becfa31/projects/rocr-runtime), with no downstream patches. The exact source was obtained into an independent task directory.

The ROCr thunk accepted `EPERM` and `ENOENT` as reasons to skip inaccessible render nodes, but Landlock returns `EACCES`. A second issue applied SVM memory policy to kernel GPU apertures excluded from the acquired user topology and used the aperture index for an architecture lookup. The task patch accepts `EACCES`, skips absent GPU IDs, and uses the corresponding user node ID. It does not change numerical kernels or weaken device access control.

The rebuilt library is preloaded into task processes only. Installed libraries, shared device permissions and external processes remain unchanged. Landlock still denies devices 0–3 before GPU initialization and applies to descendants. UUID visibility masks, HIP-to-PCI checks, and complete task-descendant DRM fdinfo audits remain mandatory. A mapped library path alone is not proof of symbol ownership: explicit SDK loading can map the installed library alongside the preloaded one.

Task root: `/workspace/validation/resume-20261106T035623Z`.

See the separate [reconstruction and public-dependency evidence](rocm-20261106-reconstruction.md) for source identity, the actual rebuilt vLLM entrypoint check, explicit native dependencies and remaining delivery work.

| Reproduction material | Task-relative path |
|---|---|
| Exact upstream checkout | `rocm-systems` |
| Source patch | `rocr-isolation.patch` |
| Build command and retained diagnostics | `build-rocr.sh`, `build-rocr*.log` |
| Task-local compiler/objcopy CMake imports | `cmake-tools` |
| Tested library and checksum | `rocr-build-v2/rocr/lib/libhsa-runtime64.so.1`, `private-rocr.sha256` |
| Strict four-device compute probe | `probe-compute.sh`, `isolation/compute.jJEnTgCR` |
| Ray actor and reordered child probe | `ray-probe.sh`, `isolation/ray.l8VYgOrY` |
| Live task-descendant audits | `audit-live.py`, `live-isolation-*.json` |

The installed wheel's full LLVM CMake export references missing unrelated executables. The task-local CMake imports bind the actual compiler and objcopy executables required by ROCr. All default image targets are built. This build and its library hash are part of the new runtime identity; prior numerical evidence is not silently reused.

## Measured progress

Updated through the November storage recovery. The table includes retained historical attempts within this task; current per-gate status is authoritative in [STATUS.md](../../../../STATUS.md). Scoped receipts do not constitute two-model release acceptance.

| Check | Result and scope |
|---|---|
| Strict GPU isolation | PASS for four-device compute, independent Ray and reordered child probes; direct kernel opens of devices 0–3 fail and task DRM handles match approved BDFs |
| Original Qwen3-8B conversion | PASS: 399 logical tensor payloads match original weights, 8,190,735,360 parameters, 36 layers |
| Independent dense math | 27 common-operator checks and 12 additional RoPE/SwiGLU/native PPO checks pass under the fixed tolerances |
| Proxy chain | Normal exit0; all three score pairs match 96 tokens per pair over theta0/theta1/theta2; three effective native Adam steps pass full-element FP64 references |
| Proxy microbatch/padding | Normal exit0; microbatch2/padding128 and microbatch1/padding64 compare equal on the recorded same-input theta0 cases |
| Full 8B controlled layout | Normal exit0 in `full8-layout-seeded.WoqHMNYP`; seeded microbatch1/padding64/cache-off versus microbatch2/padding128/cache-on has identical inputs and all64 actor/old score bytes; its three independent path pairs each match32 tokens |
| Full 8B complete gradient payload audit | All873 native gradient records are reread, hashed, shape/coverage-bound and finite: all291 native parameter tensors in each of3 seeded rollouts |
| Full 8B inference matrix | Normal exits0; 482 original-byte pairs / 1,952 compared tokens match across repeat, batch, permutation, ragged requests, prefill/decode, chunk64/16 and prefix-cache off/on; 93 real hit requests in each enabled-cache configuration |
| Full 8B unseeded chain | Theta0/theta1/theta2 preupdate checks each match all32 effective tokens across three independent paths; each backward observes all291 parameter tensors with finite gradients. The first two Adam steps pass every original parameter element, changing 6,960,402,617 and 6,957,560,633 FP32 master elements |
| Full-chain interruption | The conversation execution was interrupted during the extra third-step reference. Only26 of291 parameter references completed for that step; no normal exit is asserted. Complete checkpoints0/1 remain retained |
| Full checkpoint inspection | Seeded iteration2 has 1,421 readable DCP chunks, including complete masters/moments and RNG, scheduler steps6 and actual data sample index6. They match the fresh-process continuation in `full8-resume-comparison.5qidlhgb` |
| Full cache-training attempt | `full8-cache-micro2.HkfxeWop` exits1 during iteration0 checkpoint saving with a Ray actor RPC keepalive watchdog timeout. Its partial checkpoint is not used. A cgroup OOM-kill counter exists without attributable event timing; OOM is not asserted as the proven cause |
| Current full 8B continuation work | `full8-cache-seeded.aC392BwI` exits 0 after three native updates and complete checkpoints. All three independent score pairs match 96 tokens per pair across theta0/theta1/theta2; each deployed version starts with 0 prefix-cache tokens and its repeat hits 16. Fresh-process restore completes normally in `full8-cache-resume.QGRdzlBm`; the full continuation comparison passes |
| Full 8B new-process continuation | PASS: independent inputs/scores, all291 gradient tensors, all8,190,735,360 elements of the next native update, all1,421 checkpoint chunks, complete common state, scheduler/RNG and data progress match continuous execution; report `full8-resume-comparison.5qidlhgb` |
| Rebuilt provider entrypoint | PASS: source-reconstruction-v2 matches9,817 source files; a real vLLM proxy process executes from rebuilt providers, matches32 tokens and rejects an unsupported configuration. Native binaries are explicit dependencies; final portable packaging remains pending |
| MoE preparation | Original revision and all downloaded file identities verified: 18,867 tensors / 30,532,122,624 parameters. Full-width tie fixture passes29 independent checks; non-tie fixture passes92. Native CPU AdamW/clipping/accumulation and a small fresh-process optimizer restore pass. These do not accept a MoE model gate |
| Source reconstruction | Pinned bases plus recorded working patches reconstruct9,815 source files with matching bytes/modes. This is source reconstruction, not completed C10 model delivery |

## Execution scope and retained evidence

Training and rollout use separate physical GPUs with TP/CP/PP/DP/EP1. The initial full chain used6/7; the current seeded cache chain uses4/5. No TP4/EP4 or other multi-rank numerical result is claimed. The earlier evidence record supplies original source bases, precision and physical UUID/BDF mapping.

Original artifact paths below are relative to `/workspace/validation/resume-20261106T035623Z`:

| Evidence | Location |
|---|---|
| Full source-weight conversion | `convert-full8.vYrLldZT/logical-weight-identity.json` |
| Completed proxy score comparison | `three-path-byte-comparison.ps14wahi/comparison.json` |
| Full new-process continuation | `full8-resume-comparison.5qidlhgb/report.json`, `three-path-byte-comparison.ckvbg8dp` |
| Rebuilt actual vLLM entrypoint | `rebuilt-entrypoint.lUIjs6iJ/report.json`, exit0 |
| Full seeded layout comparison | `full8-layout-comparison.e2li_46t/report.json` |
| Complete gradient payload audit | `full8-gradient-comparison.hsiu7hp2/report.json` |
| Full inference matrix comparison | `matrix-byte-comparison.r_l5ulpe/comparison.json` |
| Initial complete-model chain and interruption | `full8.RhGRo2Mf`, including `interruption-status.json` |
| Original-byte checkpoint contents | `checkpoint-inspection/full8-iteration0.json` |
| Cache-attempt failure | `full8-cache-micro2.HkfxeWop/run.log`, exit1 |
| Explicit continuation input contract | `resume-input-contract.md` |
| Seeded cache chain | `full8-cache-seeded.aC392BwI`, exit0; `three-path-byte-comparison.fhhhvswj` |
| Full seeded checkpoint manifest | `checkpoint-inspection/seeded-continuous-iteration2.json` |
| Native MSC common-state load repair check | `checkpoint-inspection/msc-common-load.json` |
| Streaming writer and bounded optimizer reference fixture | `streaming-fixture.1AclFSxf/report.json`, exit0 |
| Non-tie MoE reference | `moe-reference-nontie.JrkwXAcH/report.json` |
| Native CPU optimizer process restore fixture | `cpu-optimizer-probe.Z0ah7Qmo`, exit0 |
| Source reconstruction | `source-reconstruction-v1/reconstruction.json` and per-repository manifests |

The initial requests omit per-request seeds. A new serving process therefore cannot be assumed to reproduce their stochastic continuation; different tokens are not compared as layout invariance. `seeded_rollout.py` explicitly submits `rollout_seed + sample.index` to native temperature-1 sampling. It preserves independent vLLM generation and the native reward/GRPO/PPO objective, and does not enable a different batch-invariant kernel provider. The uninterrupted seeded chain and its future fresh-process restore must compare the same later inputs, all gradients, the next effective update and full checkpoint state.

The continuous seeded entrypoint `launch-full8-cache-seeded.sh` has completed. The first `resume-full8-cache.sh` attempt, `full8-cache-resume.A734bTaX`, exits 1 because MSC common-state loading inherited PyTorch’s weights-only default and rejected a saved vLLM enum. The MSC branch now explicitly uses `weights_only=False`, matching the existing native ordinary-file branch for trusted full checkpoints. A direct read of the actual saved common state passes. The second restore attempt, `full8-cache-resume.Xi53Fja3`, exits 1 during redundant TE FP32 moment allocation at 189.38 GiB allocated GPU memory. The guarded native Megatron restore helper retains base Optimizer loading for plain FP32 TE state and avoids TE’s unnecessary second copy; other TE modes keep their native loader. `fp32-state-load-fixture.QqIBOooF` confirms loaded moments and the next update match original bytes, with extra allocation 8 MiB versus 0 in the fixture. The next attempt, `full8-cache-resume.15I0fbeJ`, successfully restores the full checkpoint and independently regenerates the same later samples, but exits 1 because non-straw actor creation resets the serving version to0. The actor group now initializes an unspecified publishing counter from the explicit start-rollout boundary while preserving explicit recovery versions. Eight focused actor tests pass in `version-resume-fixture.3YV6iAGP`, but `full8-cache-resume.51tl7h0Q` still exits1: serving attachment had already injected version0 before actor creation. The complete fix seeds the counter during ordinary-checkpoint resolution, before serving creation;11 focused tests including attachment now pass in `version-resume-fixture.6VFaNDVG`. The later bounded-memory continuation completed and passed, as recorded below. The source-weight audit failure is a version-binding failure, not a measured logprob mismatch. `launch-full8-layout.sh` completes on devices6/7, with `full8-layout-comparison.e2li_46t` and `three-path-byte-comparison.tn2afco3` passing. The bounded-memory restore retry on4/5 completes normally and passes its full continuation/state comparison. MoE conversion has now started after dense numerical acceptance; the generic three-rank device/collective/P2P probe passes in `pp3-device-probe.QZQUyB4n`. Do not run them concurrently with the same GPU pair or describe a prepared script as a completed restore. Final MoE engine execution, complete model gates, portable companion delivery and PR/merge remain pending. Performance: not evaluated.

## Restart storage blocker

The subsequent PP3 diagnostic attempt `moe-proxy-pp3.eJL1RKc2` again passed initial-version raw score and5,040 independent MoE field comparisons, then exited1 while writing optimizer-reference evidence because the scratch mount returned `Disk quota exceeded`. Filesystem free space (approximately33TiB) did not represent available user quota. No full MoE update, checkpoint or acceptance is claimed. One completed all-zero moment file was made sparse with identical before/after SHA256; no evidence was deleted. The [subsequent storage investigation](rocm-20261106-storage.md) binds the exact failure paths, host mounts and mapped UID1000. Container quota queries are denied, but the user-supplied host record confirms UID1000 has a 5 TiB block quota on `/dev/vdc1`, with only 2.31 GiB remaining. The existing raw-evidence workflow needs approximately 10 TiB additional headroom; a host-side adjustment to a 15 TiB limit is requested, not yet confirmed. The exact failure and retained state are recorded in `moe-proxy-pp3.eJL1RKc2/storage-blocker.json`. Both-model release and merge remain blocked.

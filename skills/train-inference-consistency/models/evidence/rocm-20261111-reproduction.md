# Reproduce the November 11 implementation

This accompanies the [acceptance record](rocm-20261111.md). It describes the delivered implementation and the exact measured environment. Prepared commands do not establish a passed gate; consult the acceptance table and original receipts. Performance: not evaluated.

## Implementation and environment

The companion is `/workspace/consistency-delivery/rc-rocm-20261111.1pwo7oqk.tar.gz`,97,905,410 bytes, SHA256 `d941e3bde7c7e3995ef277a79d67605a207aeffd15e92e9a0673a1151067f0ad`. Its293 manifest entries include pinned source archives, exact working patches/untracked files, explicit native binaries/licenses, acceptance adapters, comparison tools, fixtures and portable entrypoints. The runtime is separate from this English Markdown-only repository.

| Project | Pinned source base |
|---|---|
| vime | `d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` |
| vLLM | `98dff2a81d747d1dba01a47f939f48c3526d4206` |
| torch_memory_saver | `016938275e46e72e7b8d60e8d15046171b3e3c72` |

Measured hardware is AMD Instinct MI300X VF (gfx942), physical Devices 4–7. Use the pinned Python3.12/PyTorch2.12.0+rocm7.14.0a20260608 environment, ROCm7.14, Transformer Engine2.14.0.dev0+e6ede467, Ray2.56.1 and transformers5.10.4. The complete package freeze, public wheel identity checks and explicit binary dependencies are delivered. See the [dependency acquisition and reconstruction record](rocm-20261106-reconstruction.md). The measured reproduction scope is an isolated source reconstruction with the recorded installed Python/native ABI; it does not claim a fresh operating-system installation or independently repeated compilation of every native dependency.

The current container's `/workspace` is an ext4 bind of `/dev/vdc1[/rl-kernel/vensen/rl-consistency-validation/workspace]`. On the host, the artifact and evidence are beneath `/scratch/rl-kernel/vensen/rl-consistency-validation/workspace`. No global Docker configuration or other project directory is required.

## Restore the actual archive

Run from the measured environment, using a new destination. The restorer refuses an existing reconstruction directory and verifies every manifest entry before applying patches.

```bash
export CONSISTENCY_PYTHON=/workspace/vllm/.venv/bin/python
archive=/workspace/consistency-delivery/rc-rocm-20261111.1pwo7oqk.tar.gz
printf '%s  %s\n' d941e3bde7c7e3995ef277a79d67605a207aeffd15e92e9a0673a1151067f0ad "$archive" | sha256sum -c -
stage=$(mktemp -d /workspace/consistency-delivery/reproduce.XXXXXXXX)
tar -xzf "$archive" -C "$stage"
artifact="$stage/rc-rocm-20261111.1pwo7oqk"
"$CONSISTENCY_PYTHON" "$artifact/layout/workspace/validation/resume-20261106T035623Z/restore-delivery.py" "$artifact" "$stage/rebuilt"
export CONSISTENCY_SOURCE_ROOT="$stage/rebuilt/workspace"
export CONSISTENCY_EVIDENCE_ROOT=$(mktemp -d /workspace/validation/reproduce.XXXXXXXX)
entry="$CONSISTENCY_SOURCE_ROOT/validation/continue-20261111-831k95g3/portable-launchers"
export PYTHONPATH="$CONSISTENCY_SOURCE_ROOT:$CONSISTENCY_SOURCE_ROOT/vime:$CONSISTENCY_SOURCE_ROOT/Megatron-LM:$CONSISTENCY_SOURCE_ROOT/vllm:$CONSISTENCY_SOURCE_ROOT/torch_memory_saver"
cd "$CONSISTENCY_SOURCE_ROOT"
```

The launchers source the delivered environment, preload the delivered private ROCr, install Landlock before GPU initialization, and verify physical UUID/BDF identities. They lock both task GPU pairs and must run serially. On this host the allowed devices are4–7; relative HIP indices are not physical identifiers. The fixed UUID/BDF bindings intentionally reject a different unverified allocation.

Keep the reconstructed workspace as the working directory for `python -m` and `python -c` commands so the original workspace cannot take precedence on the Python import path. The source capture helper reads base/patch provenance from the delivered `reconstruction.json` and artifact manifest. Keep that extracted artifact directory available. It hashes actual reconstructed Python/native files and records the running source paths; it does not require the original working Git repositories.

## Original model assets and resource budget

The large original model snapshots, audited conversion and checkpoints remain in the task data mount; they are not duplicated into the source archive. Reuse these exact assets in this environment:

```bash
export CONSISTENCY_MODEL8_HF=/app/model/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218
export CONSISTENCY_MODEL8_CONVERTED=/workspace/validation/resume-20261106T035623Z/convert-full8.vYrLldZT/torch-dist
export CONSISTENCY_MODEL30_HF=/workspace/validation/resume-20261106T035623Z/model30-prepare/snapshot
export CONSISTENCY_MODEL30_PROXY=/workspace/validation/resume-20261106T035623Z/model30-proxy4
```

The full MoE snapshot is revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`; the proxy retains its first4 original-width layers,128 experts/top8 and1,575 original logical tensors. The full models retain36/48 layers and8,190,735,360 /30,532,122,624 parameters. Actual load/conversion/deployment audits compare every original logical tensor; generated text or a model directory name is insufficient identity evidence. If the dense converted checkpoint is absent, `bash "$entry/convert-full8.sh"` recreates and audits it; use its emitted `checkpoint-path.txt`. Initial model conversion has no trained optimizer state. Subsequent training and resume retain complete real optimizer/RNG state.

| Budget, before metadata/activations/runtime overhead | Qwen3-8B | Qwen3-30B-A3B |
|---|---:|---:|
| One complete model/master/moment/variance checkpoint,14 bytes/parameter |106.795GiB |398.094GiB |
| Conservative CPU working allowance for save / restore |256 /384GiB |512 /1,024GiB |
| Persistent training GPU state,18 bytes/parameter |137.308GiB on one actor |172.350 /167.134 /172.350GiB across PP3 |
| Independent BF16 rollout model |15.256GiB |56.871GiB |

These are budgets, not measured peaks or writable-quota claims. The dense original chain retains three full checkpoints and the restored next-step checkpoint; budget at least four times its per-checkpoint tensor amount if rerunning that chain from scratch, plus conversion, metadata and evidence. The MoE recipe retains one complete fork checkpoint and compares complete subsequent live-state hashes/common-state receipts. No model, master, moment or training state is removed from that fork. The delivered loader preserves native FP32 states while avoiding redundant moment allocation; the synchronous writer streams every native DCP item. The opt-in I/O memory observer records selected-device allocated/reserved/free memory and host availability around native save/load without clearing caches or changing tensors/RNG.

## Dense execution and checks

Run the full matrix, continuous chain, fresh-process resume and layout variant serially. These are reproduction commands; the current task reuses the already complete dense results after source/input compatibility checks.

```bash
bash "$entry/launch-full8-matrix.sh"
bash "$entry/launch-full8-cache-seeded.sh"
bash "$entry/resume-full8-cache.sh"
bash "$entry/launch-full8-layout.sh"
"$CONSISTENCY_PYTHON" "$entry/compare_full8_rollout_matrix.py" "$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-full8-rollout-matrix")"
"$CONSISTENCY_PYTHON" "$entry/compare_three_paths_cache.py" "$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-full8-cache-seeded")"
"$CONSISTENCY_PYTHON" "$entry/compare_full8_layout.py"
```

Inspect every chunk of the continuous and restored native `iter_0000002` checkpoints with `python -m validation.acceptance.checkpoint_manifest CHECKPOINT OUTPUT`. Create `checkpoint-inspection` beneath the evidence root and name the outputs `seeded-continuous-iteration2.json` and `seeded-restored-iteration2.json`. Then execute `compare_full8_resume.py` from the portable directory. Retain the original comparator negative controls, independent mathematical references and full gradient-coverage reports identified in the acceptance record; equality comparisons alone do not replace C6/C7.

`launch-dense-entrypoint.sh` is a shorter source-reconstruction check: it loads the full original dense model, verifies actual reconstructed providers, rejects an unsupported hidden size, restores the valid config, and compares32 original score tokens with a retained independent full-model matrix. Set `CONSISTENCY_DENSE_REFERENCE` to the accepted `full8-rollout-matrix.SBvT7Hrh/cache0-chunk64` directory beneath the November6 evidence root. It does not substitute for the continuous/resumed training acceptance.

## MoE execution and checks

```bash
bash "$entry/launch-moe-proxy-bounded.sh"
bash "$entry/launch-moe-full-bounded.sh"
bash "$entry/launch-moe-resume-bounded.sh"
bash "$entry/launch-moe-layout-bounded.sh"
bash "$entry/launch-moe-matrix.sh" full
```

The full training actor uses PP3 on Devices4/5/6; independent vLLM TP1 runs on7. TP/CP/DP/EP are1. The main chain uses microbatch1/padding64; the layout case uses microbatch2/padding128. All three score paths independently compute under the same actual weight version, masks, logical tokens and raw FP32 scoring contract. BF16 model/activation/KV bytes are retained, with FP32 main gradients/masters/moments. The repaired optimizer uses bounded GPU FP64 intermediate arithmetic and native FP32 persistent state. Independent CPU FP64 checks cover every parameter element with unchanged moment/variance2e-6, master2e-7 and update-delta0.501ULP+1e-10 accuracy criteria. These mathematical tolerances never replace original-byte score equality.

For each normally completed full/resume/layout attempt, run `compare-moe-three-paths.py ATTEMPT COUNT START`, `audit-bounded-optimizer.py ATTEMPT`, `audit-moe-boundaries.py ATTEMPT` and `audit-reconstructed-workers.py ATTEMPT`. COUNT/START are3/0 for continuous,1/2 for restored and1/0 for layout. The worker audit must receive the actual `CONSISTENCY_SOURCE_ROOT` used by that attempt. For microbatch1 continuous/resumed chains also run `moe_compact_gradient_check.py ATTEMPT`; its microbatch1 precondition must not be removed to accept layout2. Run `audit-live-weight-transitions.py ATTEMPT` on the continuous chain.

Inspect the complete full fork `iter_0000001` using `checkpoint_manifest`, naming its output `checkpoint-inspection/moe-fork-iteration1.json`. Then run `compare-moe-bounded-resume.py` and `compare-moe-bounded-layout.py`. Resume compares the actual next inputs/sampling seeds, all score/gradient/optimizer-state bytes, all final logical model tensors, native group metadata, per-rank RNG/scheduler and data cursor. Serving caches are deliberately rebuilt at fresh startup under the same weight contract.

Compare the four inference configurations using `compare_moe_rollout_matrix.py MATRIX_ATTEMPT` and `python -m validation.acceptance.moe_matrix_boundaries MATRIX_ATTEMPT`. A fresh matrix runs all four. This continuation may set `CONSISTENCY_MATRIX_REUSE_CACHE0_CHUNK64` to the explicitly completed `moe-full-rollout-matrix.h8LpVM6C/cache0-chunk64`; no other incomplete configuration is skipped. Each new engine also exercises the unsupported-config guard and verifies actual reconstructed provider imports.

The objective is the recorded native GRPO loss plus entropy coefficient0.01, with no auxiliary MoE loss or KL penalty. In the recorded equal-reward batches the policy-gradient term is zero; the entropy gradient is real, and actual nonzero gradients/master/deployed-weight changes establish effective updates. No RL reward-quality or performance conclusion is inferred.

## Native KV budget repair continuation

The original full matrix `moe-full-rollout-matrix.OIhGHJWt` completed both cache-off configurations. Its cache-on/chunk64 initialization then failed before any scored case because native vLLM calculated negative available KV memory (-99.82GiB). The original full traceback remains retained. Post-failure inspection found no task-owned GPU process and Device7 was released; complete profiler snapshots were not retained, so the record does not claim a specific accounting defect or historical external-memory event.

The explicit continuation in `moe-full-rollout-matrix.p5KC7rNY` uses the pinned vLLM's native `kv_cache_memory_bytes=2147483648` option. The sole driver change adds that constructor argument; model providers, precision, cache/chunk/batch/context settings and all112 cases/122 requests remain identical. Native startup still calls `model_runner.profile_run`; the explicit-capacity path skips automatic KV budget calculation. Actual startup reports21,840 cache tokens. At48 layers,4 KV heads, head dimension128, BF16 keys/values, one token needs98,304 bytes. Even122 requests each retaining64 tokens require only767,557,632 bytes, below2GiB. No case, context, model state or required cache behavior is removed.

The evidence bundle's `supplemental/portable-matrix-kv-repair` contains the exact executed launcher and driver. After extracting that bundle and setting `supplemental` to its `supplemental` directory, continue a failed automatic-budget matrix only when both cache-off children have complete successful receipts:

```bash
initial_matrix=$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-moe-full-rollout-matrix")
export CONSISTENCY_MATRIX_REUSE_CACHE0_CHUNK64="$initial_matrix/cache0-chunk64"
export CONSISTENCY_MATRIX_REUSE_CACHE0_CHUNK16="$initial_matrix/cache0-chunk16"
bash "$supplemental/portable-matrix-kv-repair/launch-moe-matrix.sh" full
```

The launcher requires successful exit and completed-matrix records for both reused children, allocates a new attempt and executes both cache-on configurations. Run the same four-configuration raw-score and independent MoE comparisons on its new root. Its two completed cache-off references preserve the original executed results rather than converting the failed parent into a pass. The supplemental provenance audit also compares every actual loaded weight tensor across all four configurations, verifies the three recorded reconstructed-provider/unsupported-config guards and checks the explicit native cache capacity. Final gate status remains in the acceptance record; an engine starting successfully is insufficient.

## Comparator, gradient and independent-reference commands

These CPU commands use delivered modules and allocate fresh report directories. Run the dense gradient audit after its continuous chain exits normally:

```bash
"$CONSISTENCY_PYTHON" "$CONSISTENCY_SOURCE_ROOT/validation/tooling/test_payload_compare.py" "$CONSISTENCY_EVIDENCE_ROOT"
"$CONSISTENCY_PYTHON" -m validation.acceptance.dense_gradient_report "$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-full8-cache-seeded")"
"$CONSISTENCY_PYTHON" "$CONSISTENCY_SOURCE_ROOT/validation/continue-20261111-831k95g3/actual-moe-reference.py" "$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-moe-full-rollout-matrix")"
```

The first command tests equal data, one-bit changes, signed zero, missing/duplicate/shifted entries, wrong versions, NaN/Inf and empty coverage. The last command reads the complete original model snapshot at the recorded November6 asset path and checks the first actual token in every full MoE layer using independent CPU FP64 equations. `actual-moe-backward-reference.py` in the same directory additionally replays the explicitly retained native gradient captures in `resume-20261106T035623Z/moe-proxy-pp3.eJL1RKc2`; it is a postmortem mathematical reference, not acceptance of that failed run. Its raw inputs, expert selection and exact data dependency remain recorded in `actual-moe-backward-reference.x7evh1o2`.

To rerun the independent GPU operator fixtures, use this task-isolated block after any model job has exited. It uses the same delivered source/ABI and physical identities as the native launchers; the two lock files prevent overlap with this task's model jobs:

```bash
(
  set -euo pipefail
  source "$CONSISTENCY_SOURCE_ROOT/validation/run-20261106T031920Z-devices4-7/environment.sh"
  export PYTHONPATH="$CONSISTENCY_SOURCE_ROOT:$CONSISTENCY_SOURCE_ROOT/vime:$CONSISTENCY_SOURCE_ROOT/Megatron-LM:$CONSISTENCY_SOURCE_ROOT/vllm:$CONSISTENCY_SOURCE_ROOT/torch_memory_saver"
  export LD_PRELOAD="$CONSISTENCY_SOURCE_ROOT/validation/resume-20261106T035623Z/rocr-build-v2/rocr/lib/libhsa-runtime64.so.1"
  export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
  exec 9>/scratch-tmp/consistency-gpu4-5.lock
  flock -n 9 || exit 73
  exec 8>/scratch-tmp/consistency-gpu6-7.lock
  flock -n 8 || exit 73
  export ROCR_VISIBLE_DEVICES=GPU-e07794f155d12122
  export CONSISTENCY_ALLOWED_DEVICES='[{"physical_device":7,"uuid":"GPU-e07794f155d12122","pci":"0000:bb:00.0"}]'
  export CONSISTENCY_ATTEMPT=$(mktemp -d "$CONSISTENCY_EVIDENCE_ROOT/math.XXXXXXXX")
  mkdir "$CONSISTENCY_ATTEMPT/common" "$CONSISTENCY_ATTEMPT/remaining" "$CONSISTENCY_ATTEMPT/moe-dimensions"
  export REFERENCE_DEVICE=cuda REFERENCE_ATTEMPT="$CONSISTENCY_ATTEMPT/common" CPU_REFERENCE_ATTEMPT="$CONSISTENCY_ATTEMPT/remaining"
  "$CONSISTENCY_PYTHON" "$CONSISTENCY_SOURCE_ROOT/validation/acceptance/landlock_gpu.py" "$CONSISTENCY_PYTHON" -c 'from validation.acceptance.device_guard import verify; verify("math-reference",0); import runpy; runpy.run_module("validation.acceptance.test_common_reference",run_name="__main__"); runpy.run_module("validation.acceptance.test_remaining_reference_cpu",run_name="__main__")'
  export REFERENCE_ATTEMPT="$CONSISTENCY_ATTEMPT/moe-dimensions"
  "$CONSISTENCY_PYTHON" "$CONSISTENCY_SOURCE_ROOT/validation/acceptance/landlock_gpu.py" "$CONSISTENCY_PYTHON" "$CONSISTENCY_SOURCE_ROOT/validation/continue-20261111-831k95g3/test-moe-common-dimensions.py"
  export ROCR_VISIBLE_DEVICES=GPU-99c1795a4a1184cb
  export CONSISTENCY_ALLOWED_DEVICES='[{"physical_device":5,"uuid":"GPU-99c1795a4a1184cb","pci":"0000:ab:00.0"}]'
  for fixture in test_moe_reference test_moe_reference_nontie; do
    export CONSISTENCY_ATTEMPT=$(mktemp -d "$CONSISTENCY_EVIDENCE_ROOT/$fixture.XXXXXXXX")
    "$CONSISTENCY_PYTHON" "$CONSISTENCY_SOURCE_ROOT/validation/acceptance/landlock_gpu.py" "$CONSISTENCY_PYTHON" "$CONSISTENCY_SOURCE_ROOT/validation/resume-20261106T035623Z/$fixture.py"
  done
)
```

Operator fixtures retain their existing accuracy criteria and do not replace full-model byte equality, actual backward or complete optimizer auditing. The native chains execute the independent complete optimizer reference during each real update; their audit commands above inspect its coverage and retained state.

## Resume memory repair and source compatibility

The retained first full resume `moe-resume-bounded.nYYAjIJK` loaded the complete fork, reproduced the next scores and gradients, then failed before optimizer mutation: native gradient-norm RCCL could not allocate33,554,432 bytes. Successful loading alone does not accept this run. The checkpoint observer records post-load live allocations of172.355 /167.135 /172.355GiB and allocator reserves of189.943 /185.861 /188.732GiB across the three actors.

The latest archive adds `validation.acceptance.prestep_memory.release_unused`. Only the MoE resume launcher enables `CONSISTENCY_PRESTEP_CACHE_RELEASE=1`; immediately before the unchanged native optimizer step, it synchronizes the selected device and releases unused allocator blocks. Each rank records before/after memory and asserts that live allocated bytes remain unchanged. Model, gradient, master, moment, scheduler and RNG state remain intact. Acceptance still requires a normally completed next update and complete cross-run comparison; a successful cache-release receipt is insufficient. In repair attempt `moe-resume-bounded.7MUxjxN5`, the three rank receipts show7.502 /2.924 /9.223GiB released while live allocations remain unchanged; selected-device free memory becomes9.314 /14.338 /9.164GiB. The actual container cgroup reports `memory.max=max`; the recorded current usage is a snapshot, not a save/restore peak.

The continuous full MoE chain used restored artifact `160aualb`; the actual full dense reconstruction probe used `rgogrmsc`. The latest `1pwo7oqk` archive preserves their numerical providers. Recorded compatibility checks show that removing the explicit opt-in cache guard gives the previous native optimizer callsite exactly, and the seven dense providers are unchanged. The checkpoint I/O observer is likewise an opt-in read-only wrapper around the unchanged native function bodies. See the current acceptance record for the actual restored/layout/matrix run identities; archive availability alone is not their acceptance.

## Evidence and release boundary

`/workspace/validation/continue-20261111-831k95g3/current-evidence-index.json` binds the selected original reports by SHA256. Complete raw captures and checkpoints remain at their recorded paths. Compressed row payloads are lossless, freshly observed tensors are fully compared before storage reuse, and readers validate original lengths/dtypes/hashes. The three computation branches never receive replayed scores or MoE routes from another branch.

The original native-Adam accuracy failure, cache-provenance failure, quota failure and their diagnostic/fix receipts remain preserved. Same-forward cache aliases require real native K/V byte witnesses and verified-before-consumed ordering; missing or fabricated witnesses remain failures. Prepared launchers, small fixtures, successful loading, matching text and historical environment results cannot close missing gates. Merge is conditional on both model scopes and final reproducibility evidence passing.


## Final evidence bundle and supplemental commands

The evidence bundle is `/workspace/consistency-delivery/rc-evidence-20261111.4tj6uh2a.tar.gz` (4,962,056 bytes), SHA256 `1e3c01f2e7cf786b30f8525488a1b658c8331e633560e12e1abb5523b9f65203`. Every archived member was read and checked against its manifest. It contains the exact executed native-KV launcher/driver supplement, original selected reports, retained failures and supplemental audit sources. The base runtime archive above remains unchanged. Full raw captures and model/checkpoint assets stay at their recorded task-mount paths.

```bash
evidence_archive=/workspace/consistency-delivery/rc-evidence-20261111.4tj6uh2a.tar.gz
printf '%s  %s\n' 1e3c01f2e7cf786b30f8525488a1b658c8331e633560e12e1abb5523b9f65203 "$evidence_archive" | sha256sum -c -
evidence_stage=$(mktemp -d /workspace/consistency-delivery/evidence-replay.XXXXXXXX)
tar -xzf "$evidence_archive" -C "$evidence_stage"
supplemental="$evidence_stage/rc-evidence-20261111.4tj6uh2a/supplemental"
"$CONSISTENCY_PYTHON" "$supplemental/audit-final-deployed-state.py" "$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-moe-full-bounded")" "$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-moe-resume-bounded")"
"$CONSISTENCY_PYTHON" "$supplemental/audit-checkpoint-memory.py" "$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-moe-resume-bounded")"
"$CONSISTENCY_PYTHON" "$supplemental/audit-prestep-release.py" "$(cat "$CONSISTENCY_EVIDENCE_ROOT/latest-moe-resume-bounded")"
```

The final deployment auditor checks actual serving version4 as well as the complete updated native model; the primary resume comparison separately verifies pre-update version3 and complete next-step states. `audit-matrix-reconstructed-entrypoint.py` audits the exact retained `moe-full-rollout-matrix.p5KC7rNY` reuse lineage; fresh matrices retain their own actual provider guards and full numerical/coverage comparisons. Final Markdown/shell check results, document commit, bundle identities and actual PR/merge result are bound separately in `/workspace/validation/continue-20261111-831k95g3/release-event.json`, avoiding a self-referential archive hash. Performance: not evaluated.

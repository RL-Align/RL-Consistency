# Storage investigation — 2026-11-06

This investigation continues the [November validation](rocm-20261106-resumed.md). It does not infer available quota from filesystem free space and does not accept an incomplete model run. Current gates and the bounded work schedule are recorded in [STATUS.md](../../../../STATUS.md). Performance: not evaluated.

## Historical scope and November 11 continuation

The quota values and raw-reference space budget below describe the retained November 6 failure, not a current quota query. The November 11 continuation uses complete bounded-memory mathematical checks instead of retaining redundant full-size reference arrays; native optimizer and checkpoint state remain complete. A real 427,577,923,699-byte full MoE checkpoint has now been saved and all69,193 chunks inspected. No administrator quota increase is claimed. Use the [current acceptance record](rocm-20261111.md) and [current implementation/resource budgets](rocm-20261111-reproduction.md) for the active recipe.

## Observed failure and exact paths

The failed attempt is `/workspace/validation/resume-20261106T035623Z/moe-proxy-pp3.eJL1RKc2`. At 2026-11-06 17:50:17 Asia/Singapore, the retained log explicitly reports a failed Ray log flush with `Disk quota exceeded` for:

```text
/scratch-tmp/rc-24ukkvbo/session_2026-11-06_09-45-55_068619_365000/logs/python-core-driver-01000000ffffffffffffffffffffffffffffffffffffffffffffffff_365000.log
```

Three training-worker logs independently report the same error on their respective `python-core-worker-...log` files. The driver starts a Python traceback through `native_chain_moe_pp.py`, native `vime/train.py`, and `ray.get`, but the traceback and exit-status output are truncated by the same storage failure. The observed process exit is 1. A complete original application traceback was not recovered; the investigation does not invent its missing exception frame.

Two retained optimizer files establish partial payload writes:

| Path relative to the failed attempt | Expected bytes | Actual bytes | Writer |
|---|---:|---:|---|
| `native-optimizer-reference/rank1-base0/step-000/g0-p209/variance-reference-f64.raw` | 25,165,824 | 17,752,064 | FP64 reference `files[i].write(...)`, line 68 of the captured `optimizer_reference_gpu_moe.py` |
| `native-optimizer-reference/rank2-base0/step-000/g0-p518/variance-before.raw` | 33,554,432 | 33,067,008 | `save_tensor`, called at line 43; its original-byte writer is line 5 of captured `optimizer_reference.py` |

Expected lengths follow from same-shaped retained sibling tensors and the writer's declared dtype. These paths locate interrupted application output; they are not claimed as a recovered traceback identifying which parallel rank failed first. Original logs, partial files, source snapshots and checkpoints remain intact.

## Actual storage and numeric ownership

The current mount table and the retained inspect record for container `14691cbd404d8aecc18c65fb80e88b1fe926cf42e4071d1e13bbf5a06fb829e4` agree:

| Container root | Host bind source |
|---|---|
| `/workspace` | `/scratch/rl-kernel/vensen/rl-consistency-validation/workspace` |
| `/scratch-tmp` | `/scratch/rl-kernel/vensen/rl-consistency-validation/tmp` |
| `/app/model` | `/scratch/rl-kernel/vensen/rl-consistency-validation/models` |
| `/cache` | `/scratch/rl-kernel/vensen/rl-consistency-validation/cache` |
| `/outputs` | `/scratch/rl-kernel/vensen/rl-consistency-validation/outputs` |
| `/checkpoints` | `/scratch/rl-kernel/vensen/rl-consistency-validation/checkpoints` |
| `/ray` | `/scratch/rl-kernel/vensen/rl-consistency-validation/ray` |
| `/project-config` | `/scratch/rl-kernel/vensen/rl-consistency-validation/config` |

These roots use ext4 `/dev/vdc1`, with filesystem roots under `/rl-kernel/vensen/rl-consistency-validation`. The failing files resolve to the same paths without an intervening symlink. Existing two-way host/container storage probes also bind these directories to `/scratch`. The historical `/scratch/rl-kernel/rl-consistency-ct` project was not accessed or modified.

The failure is outside the overlay container writable layer, outside its separately observed loop-backed Docker metadata storage, and outside `/dev/shm` tmpfs. `/tmp` is on overlay, but task `TMPDIR`, `TMP` and `TEMP` are `/scratch-tmp`. Ray used its recorded `/scratch-tmp/rc-24ukkvbo` directory. Hugging Face/model, pip/uv/Triton/AITER/extension caches, checkpoint and log roots resolve through the task binds above. No mount change, Docker installation/socket access or global configuration change is needed to correct these paths.

The process and files have numeric container UID/GID 0. Both namespace maps contain `0 1000 1`: these IDs map to host UID/GID 1000. The mount enables `usrjquota=aquota.user` and `jqfmt=vfsv1`. The filesystem reported approximately 33 TiB free and more than 669 million free inodes; neither number is the user's quota allowance.

Read-only `quotactl_fd(Q_GETQUOTA)` queries for user/group/project IDs 0 and 1000 return `EPERM` in this rootless, seccomp-filtered environment. This denied **query** is distinct from the observed **write** failure `EDQUOT`.

The user subsequently supplied the host output of `quota -v -u 1000`. It confirms the mapped host account `rl-kernel` has the following **user block quota**, in KiB:

| Filesystem | Used | Soft limit | Hard limit | Remaining |
|---|---:|---:|---:|---:|
| `/dev/vdc1` (task binds) | 5,366,284,580 | 5,368,709,120 | 5,368,709,120 | 2,424,540 (2.31 GiB) |
| `/dev/vda1` | 8,234,220 | 524,288,000 | 524,288,000 | 516,053,780 |

The task filesystem limit is 5 TiB, with identical soft and hard limits and no additional grace allowance. Its file count is 1,345,974 and both file-count limits are zero (unlimited). Combined with the exact failed bind paths and mapped UID, this identifies the actual storage blocker as the UID 1000 block quota on `/dev/vdc1`, not disk capacity, inode exhaustion, incorrect permissions, or a misplaced container-layer write. The 2.31 GiB remaining after the earlier sparse-file operation cannot accommodate the required complete state. The host output is preserved verbatim; it is identified as user-supplied evidence rather than a container-executed query.

## Actions and capacity budget

Two new 4 KiB probes, under the same unchanged runtime UID, passed write, fsync, readback and physical-mount checks in the task output and temporary roots. Only those explicitly named regenerable probes were removed. This proves location and permissions, not sufficient capacity for a complete checkpoint.

An earlier completed all-zero moment payload was converted to a sparse file with identical before/after SHA-256 and unchanged logical length of 2,489,319,424 bytes. Its receipt is `resume-20261106T035623Z/sparse-storage-proof.json`. No evidence bytes or checkpoint state were deleted. This recovered bounded space for records; it is not a quota bypass or proof of full-run capacity. No further broad cleanup, multi-terabyte copying or whole-dataset hashing was started.

| Complete state budget | Qwen3-8B | Qwen3-30B-A3B |
|---|---:|---:|
| Parameters | 8,190,735,360 | 30,532,122,624 |
| BF16 model + FP32 master + two FP32 moments | 114,670,295,040 bytes | 427,449,716,736 bytes |
| Actual 8B iteration-2 checkpoint, including metadata | 114,672,986,405 bytes | Not yet saved |
| Four retained full checkpoints, tensor minimum | 458,681,180,160 bytes | 1,709,798,866,944 bytes |
| Existing full raw optimizer reference, per update | 425,918,238,720 bytes | 1,587,670,376,448 bytes |

The 8B full checkpoint save and new-process restore already passed and are reused. The MoE candidate stores native BF16 model, FP32 main gradients, masters and moments on PP3 stages: approximately 172.35, 167.13 and 172.35 GiB of persistent training GPU state, before activations, communication buffers and allocator overhead. Independent TP1 rollout uses physical Device 7. Actual full-model fit remains unverified.

The MoE synchronous DCP writer copies one tensor at a time per rank instead of preloading the complete checkpoint on the CPU. Its largest embedding/head FP32 tensor is 1,244,659,712 bytes. Native restore uses the already tested bounded FP32 TE state-loading repair; a conservative 200 GiB CPU reserve is planned, but full-model peak measurement remains required. Host-wide available RAM does not prove the effective container memory limit. Neither saving nor restoration omits optimizer, RNG, scheduler or data-progress state.

With the current original-byte reference retention, three uninterrupted updates plus the restored next update, four full checkpoints, layout checks, conversion and matrix evidence require a planned **10 TiB of additional quota headroom**, including overhead. This is a workload budget, not an instruction to allocate a file of that size. Against the supplied record, the minimum calculated limit increment is 10,734,993,700 KiB (approximately 9.998 TiB). Raising the existing 5 TiB limit to 15 TiB provides the planned headroom with rounding margin.

The minimal host-side adjustment, for the administrator of the existing filesystem, is:

```bash
sudo setquota -u rl-kernel 16106127360 16106127360 0 0 /dev/vdc1
quota -v -u 1000
```

These values are KiB. The requested increase has not been confirmed or performed from this container. Recheck the resulting current usage and limit before resuming high-volume runs. No UID change, directory-based workaround, bind-mount change, Docker data-root change or action on another member's resources is required. Preserve complete optimizer/training state and the failed evidence.

## Retained investigation receipts

All investigation files are under `/workspace/validation/storage-recovery-20261106T111029Z`:

- `exact-failure-paths.json`, `retained-failure-tail.txt`, and captured failing writer sources.
- `mounts-and-small-write-probes.json`, `quota-readonly.json`, and the user-supplied `host-quota-user-provided.txt` / `host-quota-confirmed.json`.
- `checkpoint-capacity-budget.json` and `session-budget.json`.
- `moe-partial-gradient-audit.nu65jrrx/report.json`: 1,402 real gradient-hook payloads and 667 participating-expert records audited, including zero upstream padding contributions. This is four-layer postmortem backward evidence, not full-model acceptance.
- `partial-optimizer-audit.json`: 337 completed parameter references, 657,457,152 elements, all completed checks pass. Missing all-rank completion/checkpoint prevents accepting or resuming that partial update.

No PR or merge is permitted while required model and final delivery gates remain incomplete.

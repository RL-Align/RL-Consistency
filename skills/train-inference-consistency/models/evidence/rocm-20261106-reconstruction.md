# ROCm reconstruction evidence — 2026-11-06

This record accompanies the [current execution evidence](rocm-20261106-resumed.md). It establishes specific source, dependency and entrypoint checks. It does not independently accept either full model or authorize a release while required gates remain pending. Performance: not evaluated.

## Reconstructed implementation

The task root is `/workspace/validation/resume-20261106T035623Z`. `source-reconstruction-v2` rebuilds detached source trees from the following bases, applies the recorded binary-safe Git patches, and restores separately recorded untracked files. Existing working trees are preserved.

| Repository | Base commit | Reconstructed source files |
|---|---|---:|
| vime | `d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36` | 713 |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` | 2,235 |
| vLLM | `98dff2a81d747d1dba01a47f939f48c3526d4206` | 6,806 |
| torch_memory_saver | `016938275e46e72e7b8d60e8d15046171b3e3c72` | 63 |

All 9,817 source files match the corresponding working implementation in bytes and file modes. Per-repository manifests, patch hashes and untracked-file identities are retained. This check includes pre-existing environment adaptations; it does not claim every recorded patch was authored during this task.

The seven existing vLLM shared objects, totaling 187,353,040 bytes, and generated `_version.py` are explicit native dependencies, separately fingerprinted in `native-dependency-manifest.json`. They are copied into the reconstructed tree for the entrypoint check. This is not a claim that compiling vLLM from source was independently repeated. Portable delivery must include these dependencies with their licenses or provide an independently verified rebuild before C10 can close.

`rebuilt-entrypoint.lUIjs6iJ` exits 0 after loading the original-width four-layer dense proxy in a real independent vLLM process on physical Device 6. Its actual vLLM, vime, Megatron and common-provider module paths point into the reconstructed trees. Two 16-token requests match the earlier independent proxy matrix in original score bytes and semantic bindings. A deliberately unsupported hidden-size configuration is rejected before execution; the valid configuration is restored and generates the matching results. This is a representative reproduction test, not a replacement for full-model evidence.

A second real entrypoint test, `/workspace/validation/storage-recovery-20261106T111029Z/delivery-entrypoint.DXkOJAw6`, exits0 after restoring the versioned companion `rc-rocm-20261106.8xd8wqco` into a new directory. Actual provider paths point into that restored tree; the two16-token requests match original bytes and the unsupported-size guard rejects the invalid configuration. This verifies execution from the delivered source/patch layout on physical Device6, with the recorded installed Python/ROCm dependencies. It is not an independent rebuild of every native binary or a full MoE rerun. The archive SHA256 is `89e3f3102993ad8a60ed626cf199f2e2a3138f27315884a8abaa70432f362a47` (97,649,994 bytes); later package revisions require their own manifest and source compatibility check.

## Public numerical dependencies

The package checks below use actual local files and public AMD distribution contents. `dependency-acquisition` retains the queries, public RECORD files, verification scripts and reports.

| Dependency | Bound identity and acquisition evidence |
|---|---|
| PyTorch | `2.12.0+rocm7.14.0a20260608`, source `7e98855719e17e9015f9fa2e06cd4525da0b0f38`; [public CPython 3.12 Linux wheel](https://rocm.nightlies.amd.com/whl-multi-arch/torch-2.12.0%2Brocm7.14.0a20260608-cp312-cp312-linux_x86_64.whl) |
| ROCm SDK core | `7.14.0a20260608`; [public wheel](https://rocm.nightlies.amd.com/whl-multi-arch/rocm_sdk_core-7.14.0a20260608-py3-none-linux_x86_64.whl); 110 installed shared objects match its RECORD |
| ROCm SDK libraries | `7.14.0a20260608`; [public wheel](https://rocm.nightlies.amd.com/whl-multi-arch/rocm_sdk_libraries-7.14.0a20260608-py3-none-linux_x86_64.whl); 22 installed shared objects match |
| gfx942 device assets | `7.14.0a20260608`; [public wheel](https://rocm.nightlies.amd.com/whl-multi-arch/rocm_sdk_device_gfx942-7.14.0a20260608-py3-none-linux_x86_64.whl); all 1,635 hashed package entries match, including kernel packs |
| SDK development archive | `7.14.0a20260618`; [public wheel](https://rocm.nightlies.amd.com/whl-multi-arch/rocm_sdk_devel-7.14.0a20260618-py3-none-linux_x86_64.whl); archive and actual clang/clang++/objcopy identities verified |
| Triton | `3.7.0+gitb4e20bbe.rocm7.14.0a20260608`; [public wheel](https://rocm.nightlies.amd.com/whl-multi-arch/triton-3.7.0%2Bgitb4e20bbe.rocm7.14.0a20260608-cp312-cp312-linux_x86_64.whl); 13 installed shared objects match |
| Transformer Engine | `2.14.0.dev0+e6ede467`; actual installed FusedAdam Python source is byte-identical to [the pinned ROCm source](https://github.com/ROCm/TransformerEngine/blob/e6ede467a49cfda1859b145e045109e2f330bccc/transformer_engine/pytorch/optimizers/fused_adam.py); its MIT license accompanies the retained source |

The PyTorch wheel SHA-256 is `cda8f089c1c6cf9a874236d4e69845d9e22a2b4564375f9d8341658828c822f9`. All 2,205 installed Python/shared-object entries checked against that wheel match; the version file also matches. The SDK development wheel SHA-256 is `7be0c6f1a6bebf57372af4133e0beb59203df7f07373832de1dc7a7dd6f179a9`. Its archive links compiler entrypoints to SDK core binaries; this environment materializes those links as files with the same bytes. Pin both SDK dates rather than assuming every package has the same build date.

These checks establish acquisition and identity for the stated components. The complete Python package freeze, actual worker import hooks, `.pth` entries, dispatch modes and mapped-library paths remain part of each model attempt. They do not prove an untested package combination is equivalent.

## Task-private ROCr

The isolation repair uses the exact installed SDK source revision and a separately recorded patch. See the [runtime repair and physical-device evidence](rocm-20261106-resumed.md#task-private-runtime-repair). The patched library is preloaded only in task processes. Its compiler provenance is bound to the SDK checks above. Reproduction must preserve Landlock denial of Device 0–3, verified UUID/BDF mapping, relative HIP/CUDA numbering and independent Ray instances before any GPU initialization.

## Remaining delivery work

After all required full-model checks pass, assemble the final companion implementation, exact patches, native dependencies/licenses, model/fixture acquisition commands, launchers and immutable evidence index. Refresh the reconstruction manifest if subsequent repairs change source files. Validate the final Markdown references and supported-range tables against the actual receipts. Do not interpret this intermediate record as completed C10 or merge approval.

The retained build recipe names public base `rocm/primus@sha256:c3d9033c55440f3650a24b26377f3630a0b1bf1b48a6509a4b8fdd8900b2bbcb`. An authenticated read-only Docker Hub registry request retrieved its OCI manifest and config and verified both SHA256 digests; the manifest describes16.253GiB compressed layers. Receipts are `dependency-acquisition/public-base-{manifest,config,verified}.json`. This establishes availability of the named immutable base, not provenance of the running container. No Docker installation, daemon access, image pull or container replacement was performed. The source-tree rebuild and separately verified public wheels remain the direct evidence for the running implementation.

## Updated partial companion delivery

The current partial delivery is `/workspace/consistency-delivery/rc-rocm-20261106.taqvk9uy.tar.gz`,97,726,015 bytes, SHA256 `134685bd9157a94005632a48b9ddfccc913817feca5fd5cea24c7cc3fea228f1`. Its180 manifest entries include the handoff README, native-dependency licenses, exact source archives/patches, prepared launchers and ten small evidence reports bound to their original paths/hashes. Full raw evidence and checkpoints remain in place; they are not copied into this archive.

`storage-recovery-20261106T111029Z/delivery-update-verification.json` verifies every package entry and confirms that61 source/native/acceptance payload entries from the previously reconstructed and GPU-exercised package remain identical. The sole additional acceptance helper is an unimported cache-provenance candidate; its proposed runtime consumers are separately labelled under `pending-moe-cache-provenance`. A CPU fixture passes630 field comparisons and rejects wrong-version, wrong-process, future-constructor and wrong-causal-input controls. It is not actual cache-enabled MoE acceptance. The initially strict package comparison detected the added helper; that failed check and its explicit follow-up classification are retained.

The README gives the exact source restoration command and calls out remaining absolute-path and clean-environment provisioning work. Final portable environment/launch validation and full-model MoE gates remain BLOCKED. This archive is a reviewable implementation handoff, not a completed C10 release. No commit, push, PR or merge has been performed for the unfinished two-model acceptance.

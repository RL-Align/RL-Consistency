# Capturing real execution data

Two paths are available: inspect the existing RL-Kernel `.pt` sidecars, or call
the explicit trace API inside your engine workers. Neither path installs hooks
or launches a model. No GPU acceptance has been performed for this integration.

## Import existing Qwen3/Vime sidecars

Run this after the RL-Kernel job has finished writing its sidecars:

```bash
rl-consistency import-rlk \
  --source /path/to/cuda-run/mismatch-sidecars \
  --tp 4 --cp 2 --world-size 8 --layout vime-zigzag \
  --output /path/to/diagnostic.json
```

For ROCm, the upstream directory is `mismatch_sidecars` (underscore). Point
`--source` to that directory directly. Training world size includes DP replicas;
it is not the rollout engine count. Only the pinned `tp-cp-ep-dp-pp` rank order,
PP=EP=1 and Vime's non-allgather zigzag layout are supported. CP1 uses the entire
response. Sidecars do not record layout, so verify the executed configuration
before supplying `--layout`; the importer cannot discover it from tensors alone.

The importer accepts `rlkernel.vime_rocm_attention_mismatch_sidecar.v1` as
written by `tis_metrics.py` in both backend paths at RL-Kernel commit
`76a127c206b7ead6fb837b46264ba43f9ef20e82`. It checks:

- Exact rank/call coverage, contiguous calls beginning at zero, and duplicate files.
- Byte-identical TP replicas, consistent CP masks/lengths, and complete CP response ownership.
- The causal offset between target tokens and preceding logits, with no dtype conversion.
- Binary loss masks, nonempty active coverage and finite active outputs; masked rows are excluded.
- Bitwise selected-logprob equality, including signed zero, with the first logical mismatch.

The output retains normalized training/rollout raw bytes, source-file hashes,
topology and diagnostic counts. Sample IDs are derived from DP rank, call and
sample ordinal. They are **not stable input IDs**, and call indices are **not
optimizer step numbers**. Missing an entire final call or a sample consistently
on every rank cannot be detected without an external expected fixture.

Exit code `0` means equal bytes in the supplied active records, `1` means a
comparison mismatch (report retained), and `2` means malformed/incomplete data
or an I/O error. Existing output files are never overwritten. A successful
report has `certified: false` and `gate_eligible: false`. Its distinct schema
cannot be submitted as a trace or gate envelope. Model/weight version, fixture,
scoring semantics, independently executed paths and loaded implementation
identities must still be captured for formal evidence; no fake hashes are filled
in from a recipe to make a historical dump pass.

PyTorch >=2.6 must be installed explicitly in the interpreter used for import.
The loader uses `torch.load(..., map_location="cpu", weights_only=True)` and
never falls back to unrestricted pickle. Import only artifacts from your own
reviewed runs; the restricted loader is not a general sandbox. See the
[PyTorch serialization documentation](https://docs.pytorch.org/docs/stable/notes/serialization.html).
Reading JSON traces and running the normal CLI still require no PyTorch.

## Capture a formal trace inside a worker

`rl_consistency.capture.TraceWriter` accepts actual identity metadata, a worker
manifest and the **expected logical keys from the reviewed fixture**. It does
not infer the expected coverage from the observed output. The following shows
the integration boundary; variables come from the engine adapter and fixture:

```python
from rl_consistency.capture import TraceWriter, worker_manifest

# Run after loading and executing the relevant operators in this worker.
manifest = worker_manifest(
    role=worker_role,                  # "training" or "rollout"
    rank=rank,
    world_size=world_size,
    module_names=loaded_operator_module_names,
)
writer = TraceWriter(
    producer=worker_role,
    weight_version=observed_identity["weight_version"],
    weights_sha256=observed_identity["weights_sha256"],
    inputs_sha256=observed_identity["inputs_sha256"],
    semantics_sha256=observed_identity["semantics_sha256"],
    implementation_manifest=manifest,
    independent=True,                 # only after genuinely computing this path
    expected_keys=fixture_expected_keys,
)
writer.add_tokens(
    sample_id, target_token_positions, "selected_logprob", selected_logprobs,
    active_mask=active_mask,          # bool [token], or omit for all rows
)
# Repeat add_tokens for all expected samples/nodes, then publish once.
writer.finish(output_path)
```

`selected_logprobs` is a `[token, ...]` PyTorch tensor. Global **target-token**
positions and stable sample IDs are supplied explicitly; the API does not guess
causal shift, padding, packing, CP layout or data ordering. It accepts
FP16/BF16/FP32/FP64 without casting, and preserves the live autograd graph.
Expected keys are `(sample_id, target_position, node)` triples. Extra, duplicate,
missing, nonfinite or mixed-dtype records fail before publication. All-masked
batches add nothing; an empty final capture cannot pass.

Use one writer per process, weight version and dtype. Gather/align the required
outputs before writing a complete pair, or explicitly assemble disjoint shard
records with `add_records`; transport and cross-worker collection are adapter
responsibilities. Do not compare partial shards and label them full-model
coverage. `finish` atomically creates a new file and refuses to overwrite it.
Only after collection should the normal gate workflow evaluate the trace pair.

## Provenance and integration responsibilities

`worker_manifest` hashes the files of explicitly named, **already imported**
modules. On Linux it also reads `/proc/self/maps` and hashes the mapped shared
libraries. It never imports a module just to claim it was loaded. Invoke it in
every relevant worker after operator materialization. Python/module hashes and
library hashes form the implementation fingerprint; PID/rank differences alone
do not create different implementation identities. The complete worker manifest
is embedded under the trace's `capture` field.

For non-Linux development, `mapped_libraries=False` is available and recorded.
This reduced manifest is not sufficient to claim native-library coverage. Even
on Linux, a mapped file does not prove that its operator executed. Files are
hashed on disk, not from process memory; JIT-generated code, invocation counters,
operator contracts and fallbacks require engine-specific observations for C5.
Worker manifests alone are not complete C0 runtime artifacts.

The adapter must verify complete loaded parameters and synchronized weight
versions; creating a writer does not inspect weights. Training must recompute
scores from rollout tokens, rather than copying rollout logprobs. C7 still needs
an independently implemented mathematical reference. Update chains and restored
optimizer state still need their separate C8/C9 evidence.

Capture copies tensors to CPU and synchronizes device work. Use it in correctness
runs. Measure C1/C10 with matched instrumentation and declared timing boundaries;
do not compare an instrumented strict run with an uninstrumented native baseline.
This first collector holds raw records in memory and emits JSON; large intermediate
tensors need a future streaming/binary store.

## CPU validation

The standard test suite runs without PyTorch and skips the optional tensor tests.
CI also installs CPU PyTorch 2.6.0 and executes the complete suite, including real
`.pt` serialization, four dtypes, autograd preservation, CP boundaries, signed
zero, missing ranks, inconsistent replicas, masks and restricted-loader failure.
All test tensors are synthetic and provide no Qwen3 or GPU acceptance evidence.

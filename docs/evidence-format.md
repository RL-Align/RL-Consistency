# Evidence format v1

## Identities and trust

Use `rl-consistency hash FILE` for raw-file SHA256 and `hash --json FILE` for canonical JSON SHA256 (sorted keys, compact separators, finite values only). Duplicate JSON keys and NaN/Inf literals are rejected.

A recipe binds model/revision, model config, weights, tokenizer, fixture, source commits, environment lock and machine profile. Weight hashes refer to an agreed complete checkpoint/parameter manifest; both engine producers must verify that manifest against their loaded parameters. For updated versions, hash the complete canonical parameter manifest, never one sampled parameter. The loader preflight does not replace this worker check.

Environment lock canonical JSON has `python_version` and `packages` (`torch`, `vllm`, `ray`, `triton`, `transformer-engine`; use `missing` when absent). `profile_sha256` hashes the machine profile's raw bytes. Worker `loaded_artifacts` must identify actual imported source and binary files, including accelerator extensions. Record one entry per relevant worker/artifact, not just rank 0's launcher.

Gate evidence declares `execution: real`; validators cannot independently prove that external GPU producers executed. Raw producer code, worker manifests and logs must be audited. Synthetic test fixtures are never certification evidence.

## Gate envelope

```json
{
  "schema": "rl-consistency.evidence.v1",
  "gate": "C6",
  "recipe_sha256": "<canonical bound recipe SHA256>",
  "environment_sha256": "<canonical recipe.environment SHA256>",
  "execution": "real",
  "pairs": [{"case": "train-rollout", "left": "rollout.json", "right": "training.json"}]
}
```

Paths must resolve under the envelope's directory. Files are copied into the run; later changes to the original producer directory do not mutate the snapshot. Checkpoint nested trace paths use the envelope directory too. A different filename alone does not make computation independent: producers must actually run both paths.

## Trace

```json
{
  "schema": "rl-consistency.trace.v1",
  "producer": "rollout",
  "independent": true,
  "weight_version": "step-0",
  "weights_sha256": "<complete parameter/checkpoint manifest hash>",
  "inputs_sha256": "<recipe inputs.fixture_sha256>",
  "semantics_sha256": "<canonical recipe.comparison SHA256>",
  "implementation_sha256": "<actual implementation manifest hash>",
  "dtype": "float32",
  "records": [
    {"sample": "sample-001", "position": 12, "node": "selected_logprob", "shape": [], "hex": "000080bf"}
  ]
}
```

The illustrated bytes represent -1.0 in little-endian FP32; they are an encoding example, not a test result. `shape: []` is a scalar. Every logical `(sample, position, node)` must occur exactly once. Shapes and token/node coverage must match across each pair. Traces must be finite and nonempty. Supported output dtypes: `float16`, `bfloat16`, `float32`, `float64`. `compare.tensor_record` is an optional PyTorch capture helper; it does not remove padding or assign model semantics for you.

The comparator's `first_mismatch` is in sorted logical-key order; it is not a topological first-divergence locator. Use the model graph and intermediate captures for causality.

The optional `capture.TraceWriter` API validates complete fixture-declared key
coverage and atomically publishes traces. Its `capture.worker_manifest` records
loaded-file fingerprints; this does not replace the complete runtime/model
artifact for C0 or observed operator bindings for C5. See [capture.md](capture.md).
The historical sidecar importer uses a separate `rl-consistency.rlk-diagnostic.v1`
schema and does not generate gate-eligible trace identities.

For C7, use producer `training` versus `reference`, different implementation fingerprints and `reference_kind: independent_mathematical`. Use the recipe's pre-reviewed `atol`/`rtol`; tolerances in result files have no effect. Both mandatory cases `forward-accuracy` and `gradient-accuracy` must be present. Declare the complete tensor/parameter set being tested in the fixture and review it; the generic toolkit cannot infer unrecorded model parameters.

## Gate-specific artifacts

| Gate | Envelope fields | Referenced contents |
|---|---|---|
| C0 | `runtime` | `clean`, `rl_kernel_commit`, `sources`, `lock_sha256`, `profile_sha256`, `python_version`, `backend`, `architecture`, `world_size`, `training`, `rollout`, `model`, `fixture_sha256`, `loaded_artifacts` |
| C1 | `benchmark` | Native benchmark object below |
| C2 | `coverage` | `source_config_sha256`, `source_layers`, `selected_layers`, `reduced_layers`, `boundary_operators` |
| C3/C4/C6/C7 | `pairs` | Exactly the case IDs declared in recipe coverage |
| C5 | `runtime`, `bindings` | Runtime identity and binding list below |
| C8 | `pairs`, `updates` | Post-update pairs and update records below |
| C9 | `pairs`, `full_model`, `checkpoint` | Full-model and resume records below |
| C10 | `benchmark` | Strict benchmark object; frozen C1 is loaded from the run |

`loaded_artifacts` entries contain `worker` (`training` or `rollout`), `path`, `sha256`. The full target/model objects equal their recipe objects. Runtime artifacts must include both engine roles. Implement full rank coverage in the producer; maintainers verify it.

C2 source layers are ordered objects with `operators`, `dimensions`, `precision` and any layer-specific semantics (e.g. attention type). `reduced_layers` must equal the selected source layer objects; do not include a physical layer index inside a semantic layer descriptor. All operator types and adjacent semantic layer patterns must remain. `boundary_operators` equals the recipe operator inventory and includes embedding/head. The coverage object is an assertion backed by the builder's source; it is not itself proof of executing those layers.

C5 bindings are a list with one entry per `(side, operator)` required by the recipe. Fields: `side`, `operator`, `observable: true`, `fallback: false`, `implementation_sha256`, `contract_sha256`. Both sides must share each operator's numerical contract, but may use distinct validated implementations. Add trace provenance/call counters in the real producer to establish that registered operators were actually invoked.

C8 updates are a list of at least two records: `version`, `before_sha256`, `after_sha256`, `optimizer_state_sha256`, `gradient_nonzero: true`, `optimizer_step_executed: true`. The chain begins at recipe model weights and links consecutive versions. Each updated weight hash/version must have an independently computed parity pair in the evidence. Record complete parameter manifests and raw optimizer traces alongside these assertions for review.

C9 `full_model` contains `config_sha256`, `layers`, `context_tokens`, `target`; it must match the unmodified model and cover the declared maximum context. `checkpoint` contains:

- `uninterrupted_state` and `restored_state`: matching SHA256 for `model`, `optimizer`, `scheduler`, `rng`, `data_cursor`.
- `resumed_first_step: true`.
- `uninterrupted_trace`, `resumed_trace`: independently captured first-step trace paths.
- `uninterrupted_after_update_sha256`, `resumed_after_update_sha256`: matching complete parameter hashes after the first resumed update.

The separate `checkpoint-resume` pair in C9 verifies train/rollout parity of that resumed version. Resume/control equality and train/rollout parity are two different tests.

## Benchmark object

Fields: `mode` (`native`/`consistency`), `fixture_sha256`, `model_sha256`, `target` (recipe target), `settings` (recipe benchmark), `sampling` (recipe sampling), `execution_sha256`, `warmup_steps` (at least 1), `step_seconds` (at least 3 finite positive values), `total_output_tokens`, `peak_memory_bytes`.

C10 compares replayed equal work and requires equal measured step/output-token counts. It reports mean step overhead and memory ratio, without promising a speedup. Sampling, loss, input content, memory definitions and timing boundaries must be included in the reviewed fixture/settings. This first comparator is not a statistical benchmark suite.

## Review records

After a real human decision, record it with `rl-consistency review baseline --run-dir RUN --reviewer NAME --note DECISION` (replace `baseline` with `precision` or `release` when appropriate). The baseline binds C1, precision binds C0, release binds all gates. The command does not authenticate identity. Never run it with an invented approval. Changing numerical tolerances requires a new run and review.

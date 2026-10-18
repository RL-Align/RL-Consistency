# Evidence format v2

Qwen3.8-27B adds model-specific requirements to this v2 format. C0, C2 and C9
must reference the actual full HF JSON as `model_config`; its canonical hash
must match `recipe.model.config_canonical_sha256`. C2 also references
`proxy_config`. Its coverage must match the adapter-derived layer descriptors
and selected prefix, with `checkpoint_materialized: true` and a real
`weight_manifest_sha256`. A `prepare-model` bundle is not acceptance evidence.
Keep the Hybrid C3/C7/C9 cases in the recipe; see the
[model guide](../skills/train-inference-consistency/references/qwen38-27b.md).

Qwen3-30B-A3B uses the same actual-config and materialized-proxy requirements.
It additionally requires independent `routing_pairs` at C3/C5/C8/C9, and separate
uninterrupted/resumed routing captures at C9. Read [MoE routing evidence](moe-routing.md)
and keep the MoE-specific C3/C7/C9 cases; output equality alone is insufficient.

Recipe, evidence and run schemas are v2. Trace encoding remains v1.
V1 recipes/evidence are rejected;
create a new run and supply the additional artifacts rather than relabeling old
results. The [gate table](../skills/train-inference-consistency/references/gates.md)
and [workflow boundaries](../skills/train-inference-consistency/references/workflow-scope.md)
define the scope.

## Identities and trust

Use `rl-consistency hash FILE` for raw-file SHA256 and `hash --json FILE` for canonical JSON SHA256 (sorted keys, compact separators, finite values only). Duplicate JSON keys and NaN/Inf literals are rejected.

A recipe binds model/revision, model config, weights, tokenizer, fixture, source commits, environment lock and machine profile. Weight hashes refer to an agreed complete checkpoint/parameter manifest; both engine producers must verify that manifest against their loaded parameters. For updated versions, hash the complete canonical parameter manifest, never one sampled parameter. The loader preflight does not replace this worker check.

Environment lock canonical JSON has `python_version` and `packages` (`torch`, `vllm`, `ray`, `triton`, `transformer-engine`; use `missing` when absent). `profile_sha256` hashes the machine profile's raw bytes. Worker `loaded_artifacts` must identify actual imported source and binary files, including accelerator extensions. Record one entry per relevant worker/artifact, not just rank 0's launcher.

Gate evidence declares `execution: real`; validators cannot independently prove that external GPU producers executed. Raw producer code, worker manifests and logs must be audited. Synthetic test fixtures are never certification evidence.

## Gate envelope

```json
{
  "schema": "rl-consistency.evidence.v2",
  "gate": "C5",
  "recipe_sha256": "<canonical bound recipe SHA256>",
  "environment_sha256": "<canonical recipe.environment SHA256>",
  "execution": "real",
  "pairs": [
    {"case": "old-logprob", "left": "rollout-old.json", "right": "old-logprob.json"},
    {"case": "actor-forward", "left": "rollout-actor.json", "right": "actor-forward.json"}
  ],
  "runtime": "runtime.json",
  "bindings": "bindings.json"
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

For C7, use producer `training` versus `reference`, different implementation fingerprints and `reference_kind: independent_mathematical`. Use the recipe's pre-reviewed `atol`/`rtol`; tolerances in result files have no effect. Cases `forward-accuracy`, `gradient-accuracy` and `optimizer-accuracy` must be present. Declare the complete tensor/parameter set and gradient/STE contract in the fixture; the generic toolkit cannot infer unrecorded parameters. Controlled SFT and generation quality remain separate, unimplemented acceptance items.

## Gate-specific artifacts

| Gate | Envelope fields | Referenced contents |
|---|---|---|
| C0 | `runtime`, `inventory` | Worker identities plus complete module inventory and training route |
| C1 | `benchmark` | Native benchmark object below |
| C2 | `coverage` | `source_config_sha256`, `source_layers`, `selected_layers`, `reduced_layers`, `boundary_operators` |
| C3/C7 | `pairs` | Exactly the case IDs declared in recipe coverage |
| C4 | `profile` | Structural profile artifact below |
| C5 | `runtime`, `bindings`, `pairs` | Runtime bindings plus independent old-logprob/actor forward traces |
| C6 | `backward` | Backward inventory, saved state, contract and measured training budget |
| C8 | `pairs`, `updates`, `loop` | theta0/theta1/theta2 comparisons and two effective master updates |
| C9 | `pairs`, `updates`, `loop`, `full_model`, `checkpoint` | Reviewed full-model loop, dual-path entry and resume |
| C10 | `benchmark` | Strict benchmark object; frozen C1 is loaded from the run |

`loaded_artifacts` entries contain `worker` (`training` or `rollout`), `path`, `sha256`. The full target/model objects equal their recipe objects. Runtime artifacts must include both engine roles. Implement full rank coverage in the producer; maintainers verify it.

`runtime` contains `clean`, `rl_kernel_commit`, `sources`, `lock_sha256`,
`profile_sha256`, `python_version`, `backend`, `architecture`, `world_size`,
`training`, `rollout`, `model`, `fixture_sha256`, `loaded_artifacts`.
`inventory` contains `route` equal to the training engine, and one `modules`
entry per recipe operator, with `operator`, `inference_implementation`,
`training_math`, `quantization`, `discrete_choices`. Explicitly write `none`
where applicable; do not leave unknown semantics blank.

C2 source layers are ordered objects with `operators`, `dimensions`, `precision` and any layer-specific semantics (e.g. attention type). `reduced_layers` must equal the selected source layer objects; do not include a physical layer index inside a semantic layer descriptor. All operator types and adjacent semantic layer patterns must remain. `boundary_operators` equals the recipe operator inventory and includes embedding/head. The coverage object is an assertion backed by the builder's source; it is not itself proof of executing those layers.

C5 bindings are a list with one entry per `(side, operator)` required by the recipe. Fields: `side`, `operator`, `observable: true`, `fallback: false`, `implementation_sha256`, `contract_sha256`. Both sides must share each operator's numerical contract, but may use distinct validated implementations. Add trace provenance/call counters in the real producer to establish that registered operators were actually invoked.

C5 and C9 training traces for `old-logprob` and `actor-forward` contain an
`execution_path` object: `kind` equal to the case, `forward_executed: true`,
positive `calls`, `callsite_sha256`, `model_mode` (`train`/`eval`),
`grad_enabled` and `output_requires_grad` booleans. The actor path requires
`train` mode and both grad flags true. Preserve the actual native old-logprob
path's modes, rather than forcing grad on. Capture logprobs before loss transforms
at theta0. Pass the observed receipt through `TraceWriter(execution_path=...)`.

C4 `profile`: `raw_profile_sha256`, full `target`, `per_token_python_loop: false`,
`unbounded_workspace: false`, `graph_capture_replay: true`, positive
`peak_memory_bytes`, and `operator_shapes` entries (`operator`, positive `shape`
dimensions, positive `calls`) covering the operator inventory. Keep the raw
profiler output for review; booleans do not prove that profiling happened.

C6 `backward`: `modules` contains one entry per operator with `strategy`
(`reuse`/`glue`/`custom`), `rationale`, `saved_from_forward` and
`gradient_contract_sha256`. `timing` contains full `target`, finite positive
`native_step_seconds` and `strict_step_seconds`. The ratio must be within
the human-reviewed recipe `acceptance.training_step_ratio_max`; it is unbound
in candidate recipes, not a guessed performance promise.

C8/C9 `updates` has exactly two records: `version`, `before_sha256`,
`after_sha256` (deployment weights), `before_master_sha256`,
`after_master_sha256`, `optimizer_state_sha256`, positive finite
`master_update_norm` and `gradient_norm`, `finite_loss: true`,
`gradient_coverage_complete: true`, `optimizer_step_executed: true`, and
`optimizer_step` 1 then 2. Master hashes must change and form a chain starting
at `recipe.model.master_weights_sha256`. Deployment hashes may remain unchanged
under quantization. The `theta-0` pair uses initial weights, then `theta-1` and
`theta-2` must match the synchronized versions and deployment hashes. `loop`
contains `normal_exit: true`, `executed_updates: 2`, `framework_entrypoint`.
Retain actual parameter/optimizer evidence to back all these assertions.

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

After a real human decision, record it with `rl-consistency review baseline --run-dir RUN --reviewer NAME --note DECISION` (replace `baseline` with `precision`, `recipe` or `release` when appropriate). Baseline binds C1; precision binds C0; recipe review binds C0/C5/C7/C8 and is required before C9; release binds all gates. The command does not authenticate identity. Never invent approval. Changing numerical tolerances or the training budget requires a new run and review.

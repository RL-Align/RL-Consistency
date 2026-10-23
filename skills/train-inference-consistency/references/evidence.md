# Evidence and target-project capture contract

Implement capture/comparison tooling in the target repository using its language and framework. This skill prescribes evidence semantics, not a required Python API, JSON schema or launcher. Keep Markdown summaries readable and raw tensors in lossless binary artifacts; create machine-readable indexes when needed for complete coverage checks.

## Bind actual execution

For every run and worker, record the model revision/config, complete master and deployed weight manifests/version, tokenizer and fixture hashes, source refs plus uncommitted diff, actual loaded modules/extensions, build/runtime/library versions, hardware identity, numerical contract, engine flags and relevant process groups. A checkout SHA does not prove what a worker imported. A config match does not prove checkpoint provenance.

For a capture, retain:

| Field | Required meaning |
|---|---|
| Producer/run/attempt/worker identity | Unique independently executed source, role, device/rank and callsite |
| Phase and mode | Rollout, independent old-logprob or actor; train/eval, autograd, autocast, cache and graph modes |
| Logical key | Stable sample/request, token position, layer/node, port, shard/expert/slot where relevant |
| Input semantics | Exact token IDs, masks, positions, packed boundaries, prefix/cache ownership and score definition |
| Weights and state | Synchronized policy version, master/deployment identity; necessary KV/recurrent/RNG/router state |
| Tensor descriptor | Original dtype, dimensions, layout/shard mapping, byte order, payload path/length and digest |
| Coverage | Predeclared expected entries and observed entries, duplicate/missing counts, validity rules |
| Observation limits | Opaque fused regions, partial state, observer-induced path changes and unavailable ranks |

Record expected coverage before execution. Independently execute each side/repeat; never duplicate a payload to supply a missing result. Record actor output `requires_grad` and evidence that the actual training callsite ran. Capture hooks must not replace outputs, detach the actor graph or alter routing/cache behavior. Measure whether observation disables fusion or changes dispatch, and validate the claimed final route after removing intrusive instrumentation.

Carry the producer's stable sample ID through the actual batch builder, DP partition and microbatch fields. A local partition index is a position in the current rollout, not necessarily that sample ID. They can coincide at theta0 and diverge at theta1, or after reordering/resume. Preserve both fields when useful, and test a case where they deliberately differ. Validate identity and coverage before an optimizer advances the policy version; do not repair a mismatch by renumbering captured rows without an observed mapping.

## Comparison procedure

1. Validate provenance and scope compatibility, including synchronized weights and scoring semantics. If incompatible, report blocked/incomparable and resolve the mismatch.
2. Validate complete expected coverage, unique keys, masks, nonempty valid outputs and finite tensors. Missing evidence never becomes zero mismatches.
3. Align logical entries before comparing dtype, shape and raw bytes. Preserve both original payloads and mismatch locations/counts. Use dtype-aware metrics such as max absolute/relative error or ULP only as additional diagnostics.
4. Keep raw rollout/old/actor and repeat captures distinct. Retain the exact fixture and execution context needed to reproduce the failure.
5. For parallel branches, interpret differences through the semantic dependency DAG; a sorted list's first row is not necessarily the earliest causal location.
6. Write a comparison artifact with its inputs/digests, expected/observed counts, verdict, exclusions and limitations. Reference it from the gate record rather than manually asserting pass.

Logprobs are the minimum train/rollout output; intermediate boundary evidence establishes localization and coverage. MoE also needs independently computed router choices/weights and dispatch/combine mappings; stateful models need equivalent logical state. A tolerance-based accuracy check must be stored separately from bitwise results.

## Retention and failure

Use new attempt directories and append records. Preserve nonzero process exit, timeouts, non-finite values, unsupported-provider errors and partial traces. A command that did not complete normally cannot satisfy an end-to-end gate. Verify artifact hashes before resuming or publishing; hashes detect drift, not authenticated execution against a malicious writer.

Human decisions must reference actual evidence and their source. Do not fabricate reviews, binary attestations or signatures. A Markdown template remains pending until populated by real work; even fully populated prose requires raw artifacts and inspection.

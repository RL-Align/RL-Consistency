# Extend the skill to another model

Use the common C0–C10 procedure for any model family. A new Markdown guide records architecture knowledge and required cases; it does not implement kernels or certify the model. Put executable implementation, capture adapters, reference code and tests in the target projects. Default to [correctness-first.md](correctness-first.md): implement simple common arithmetic where necessary and leave performance work optional.

## Discover before specializing

Read the actual checkpoint config, tensor index and loaded training/rollout implementations. Identify parameter mapping, every layer family, state/cache ownership, objective/scoring semantics, parallel partitions, quantization and discrete choices. Do not infer these from a marketing name or a related checkpoint. Separate text, vision, audio, multimodal projector, MTP and speculative paths as applicable; excluded paths must remain excluded from claims.

Copy [model-profile.md](../templates/model-profile.md), then classify each module:

- Covered by an inspected provider with the same dimensions and numerical contract: reuse and verify dispatch/bindings.
- Mathematically compatible but numerically different: implement contract/glue or a compatible provider and validate casts/reductions.
- New operator, state model, quantization or distributed behavior: implement it and its backward/reference/capture work before claiming support.
- Unobservable or unsupported in the target engine: record the blocker and smallest reachable diagnostic boundary.

## Define the extension work

| Work item | Required output in the target project |
|---|---|
| Architecture and weights | Validated config/tensor mapping, tied/shared/expert weights and actual load checks |
| Proxy | Source-preserving layer selection and materialized weights covering every required family/interaction |
| Forward contract | Selected or implemented numerical providers, explicit dispatch predicates and rejection of unsupported fallback |
| Backward | Per-module gradient/saved-state mapping, distributed scale semantics and independent reference tests |
| Observability | Logical token/node/state/router mappings on real rollout, old-logprob and actor callsites |
| Validation matrix | Shape/batch/context/cache/topology/precision cases, routing/state edge cases and coverage expectations |
| Delivery | Actual commands, evidence, supported ranges, limitations and model-specific reproduction notes |

## Add the Markdown profile

Place one guide in `models/`, link it from `SKILL.md` and update the README support table. Describe differences from the common workflow rather than copying all C0–C10 instructions. Make architecture facts conditional on inspected config/revision and record unresolved variants. Keep specific dependency versions as target-run identities, not universal skill prerequisites.

Exercise the guide on a bounded task-local fixture and inspect whether another agent can choose the correct paths, proxy and missing evidence without conversation history. Synthetic/tiny fixtures test the procedure; a full model requires its own C9 evidence. Any verified claim must name model/revision, hardware/software, engines, dtype, topology, input ranges, evidence and review source. Otherwise report guide/implementation availability and the remaining gates.

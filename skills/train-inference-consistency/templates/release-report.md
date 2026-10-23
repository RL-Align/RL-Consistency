# Consistency delivery report

Model/revision and target repositories: TODO. Acceptance mode: consistency-only by default. Delivery status: incomplete until evidence is verified.

## Delivered implementation

List real changed files/commits, providers/glue, capture, independent references and regression tests. Give the actual setup and reproduction commands, artifact locations and identity verification procedure. State dependencies of the target implementation; the Markdown skill itself supplies no runtime.

## Scope and model support

| Model/revision | Guide available | Implementation/ref | Hardware/software/engines | Dtype/topology/batch/context/cache | Measured acceptance and evidence | Exclusions |
|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | not-started | TODO |

## Gate results

Link the complete C0–C10 ledger and immutable attempts. Summarize pass/fail/blocked/stale stages and C1/C4 not-required when applicable, retained mismatches and missing coverage. Report rollout↔old and rollout↔actor separately, with independent routing/state where applicable. Record effective master updates, full-model cases and uninterrupted/resumed first forward/update results.

## Accuracy and performance

Record independent forward/gradient/optimizer references, tolerances, observed errors and limitations. In consistency-only mode write performance **not evaluated**; no benchmark, profile or speed target is required for delivery. Slow execution is acceptable when required checks actually complete.

Only when performance was requested, identify the frozen native baseline and equal-work configuration, warmup, repeated raw timings, variability, throughput/step cost and identical memory semantics. Explain overhead from measured profiles. Keep RL reward/quality results separate and do not infer them from parity.

## Reproduction and decisions

State reproduction actually performed, source/binary/model/fixture bindings, unsupported-route behavior, and actual precision/recipe/release decisions (and baseline decisions only when enabled) under the target policy. List unresolved blockers and the next executable action. If the user requested a subset or hardware was unavailable, identify that scope explicitly; do not label the report full-model verified or acceptance complete. When every required correctness stage passes, report “C0–C10 consistency-only acceptance complete; C1/C4 not-required; performance not evaluated”. Do not claim that all eleven stages passed.

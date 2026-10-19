# Diagnose models without strict-runtime adaptation

Use the standalone `diagnostics` API before recipe onboarding when the user asks
to locate native drift on a model absent from the registry. Read the repository's
`docs/diagnostics.md` for the exact API and formats. This route needs neither an
RL-Kernel operator substitution nor a fabricated candidate recipe.

1. Inspect native engine/model support and the actual worker execution. Use
   `discover_modules(model)` for an inventory, then create an explicit semantic
   boundary mapping. Names and shapes alone never establish correspondence.
2. Pin weights/version, canonical tokens, masks, positions and state fixture.
   Declare every expected sample/call/shard boundary and its dependencies before
   capture. Keep rollout/old-logprob and rollout/grad-actor as separate cases.
3. Capture both native implementations with `DiagnosticWriter` or explicit
   `ModuleCapture` bindings. Preserve dtype, logical slices and independent
   computation. Record actual worker provenance and observer effects. Default
   module state is partial; include KV, RNG, router and engine state before
   asserting completeness. Unknown fused interiors stay opaque.
4. Reset the same state and independently repeat each side. Save four separate
   capture IDs/directories. Do not copy a capture to supply the repeat.
5. Run `rl-consistency diagnose --matrix <manifest.json> --output <new-directory>`.
   Exit 1 is an observed mismatch; exit 2 is incomplete/incomparable/invalid
   evidence. Inspect repeatability before attributing a stable cross-path cause.
   Read the expandable HTML cells and retain JSON plus all source tensor blobs.
6. Follow the DAG frontier, which may contain multiple parallel nodes. If an
   ancestor is missing, fill the gap first. Refine opaque regions with a new plan
   and new captures. Do not bisect under a monotonic-drift assumption or use
   lexicographic names/cross-rank wall clocks as execution order.
7. For each candidate, use `replay_boundary` with original provider callables,
   an actual reset/state callback, and the original left boundary input bytes.
   Preserve physical batch/layout/precision/engine context in the adapter. The
   helper runs each endpoint twice and binds the receipt to original captures.
   `local_replay_mismatch` applies to that replay context; confirm on the
   original execution path before attributing the production cause.
8. Report the observed frontier, missing/opaque coverage, input/state changes,
   repeat instability, replay result and next boundary to instrument. Preserve
   regression evidence and re-run the affected production cases after a fix.

For CPU tooling calibration, `rl-consistency diagnostic-demo --output <new-dir>`
injects synthetic intermediate drift that reconverges before final output. It
requires optional PyTorch and intentionally exits 1. Do not relabel its outputs
as evidence for Qwen or any real model.

Do not reject an unknown model merely because `prepare-model` cannot register
it. Complete inventory, mapping and native capture integration where possible.
When an engine cannot expose a boundary, report the smallest observed interval
and the missing adapter work. Never claim automatic vLLM/Megatron instrumentation,
all-model consistency, production root cause or C0–C10 acceptance from this tool.

# C2 — Build a source-preserving representative proxy

**Entry:** C0 model/operator inventory and actual source weights. A tiny randomly initialized architecture is a tooling fixture, not this proxy.

## Construct and verify

1. Enumerate layer families, dimensions, precision, operators, tied/shared paths and state interactions from the real model. Include embedding, final norm/head and any boundaries required by the comparison scope.
2. Select a reduced depth that covers the operator families and relevant ordered transitions. A contiguous prefix is a useful starting point, not a universal rule. Usually several layers suffice; expand it or add declared fixtures when transitions, periodic layers, MoE patterns, state or sharding requirements demand more.
3. Preserve per-layer widths, head geometry, expert counts/top-k, convolution/state sizes, normalization conventions, positions, quantization and topology-sensitive shapes. Do not shrink dimensions to fit hardware and then transfer that result to the original kernel shapes.
4. Materialize proxy weights from the selected original tensors. Preserve values/dtypes, names/mappings, tied weights and selected layer identities. Keep source weights immutable. Verify the materialized manifest against the source and actual framework loaders; a generated config alone is insufficient.
5. Define expected operator/transition coverage and case matrix before testing. Retain a source-to-proxy map with evidence for every required feature; list excluded multimodal/MTP/speculative paths explicitly.
6. Run actual proxy load/forward probes and inspect executed providers/shapes. Catch missing or randomly initialized weights, unsupported layer schedules and converter omissions. Add target-project regression tests for the selection/materialization procedure.

## Model-specific selection

For a Hybrid L/L/L/F pattern, one four-layer cycle misses F→L; use at least two complete cycles if that is the observed schedule. For a uniform MoE prefix, four layers may cover the repeated family, but preserve all experts/top-k and add routing/state/distributed edge cases. For unfamiliar patterns, derive the minimum from coverage, not the model name.

## Artifacts and exit

Retain source/proxy configs, selected-layer map, real tensor manifests/digests, coverage table, load results and actual operator traces. Pass only when required features and transitions are represented with original layer semantics and weights. Keep full-model long-range/depth/resource effects for C9.

If memory or provider support prevents a representative proxy, mark that requirement blocked, complete smaller explicitly labeled diagnostic work, and report the missing coverage. Do not weaken the full-model acceptance contract to fit the proxy.

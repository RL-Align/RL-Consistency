# Qwen3.8-27B Hybrid Dense development guide

Treat `Qwen3.8-27B` as the requested checkpoint label and inspect its actual revision/config/weights. This repository's intended profile is a Hybrid Dense text path with nested `qwen3_5`/`qwen3_5_text` configuration. If the real checkpoint differs, use the generic extension procedure and update the profile; do not infer identity or rename a checkpoint solely from shared architecture code.

Start with a declared text-only scope unless the user requests multimodal execution. Explicitly account for or exclude vision/video encoders, projectors, MTP and speculative paths; text results do not verify them. Full-model GPU acceptance remains to be performed in the target environment.

## Inventory and proxy

Inspect the actual text layer schedule and every full-attention, linear-attention and convolution/state dimension. For the expected repeating L/L/L/F schedule, select at least two complete cycles (eight layers) so the proxy covers L→L, L→F and F→L. If the schedule differs, derive a new selection from actual transition coverage. Preserve widths/head counts/state sizes and the selected original weights.

Materialize embedding, selected language layers, final norm/head and exact converter mappings. Preserve source config fields and state the excluded execution paths. Check actual weight loading without silently initializing missing tensors. A proxy layer list or retained dormant vision config is not proof of running that path.

## Numerical boundaries to implement

| Boundary | Inspect and align |
|---|---|
| Common Dense | Norm scale convention, including possible `1 + weight`, epsilons, residual casting, gated MLP and GEMM |
| Full attention | Q/K normalization, partial/multiaxis rotary positions, output gate, KV precision and prefill/decode order |
| GDN projections/gates | Q/K/V/Z/A/B mappings, grouped head expansion, sigmoid/softplus/decay construction and casts |
| Short convolution | Width, causal history ordering, activation, packed boundaries, reset and decode update |
| GDN recurrence | Q/K normalization, query scaling, decay and delta update order, chunk/recurrent equivalence and state precision |
| Gated norm/output | Norm/gate order, learned scale convention, output projection and rounding |
| State ownership | Convolution history, recurrent matrix and KV by request/layer/position and weight version |
| Backward/parallelism | Saved recurrence/gates, convolution gradients, replicated/sharded gathers and reduction/scale semantics |

Inspect the selected provider and its device support. An optional acceleration package can change dispatch; bind actual loaded functions and exercise the route. Similar-looking norm replacements or chunk/recurrent formulas may change semantics or rounding. Do not assume a CUDA-specific provider works on ROCm.

## Extend the common matrix

At C3 include chunk-versus-recurrent processing, convolution/recurrent resets, packed request boundaries, cache reorder/reuse, prefix reuse and equivalent logical state. Compare states only after establishing their semantic correspondence; equal byte shape alone is insufficient.

At C5 observe the actual training and rollout implementations, plus independent old-logprob and actor execution. Following [correctness-first.md](../references/correctness-first.md), a common sequential recurrence with actor autograd is a valid final choice; optimized chunk/scan evaluation is optional. Preserve required cache/state interfaces and test their transitions. An uncached full-prefix implementation does not establish acceptance of an explicitly required cached path.

At C6/C7 independently validate projection/gate math, convolution, recurrence, gated norm and parameter/input gradients. A recurrence-only reference cannot validate the whole block. At C8/C9 rebuild or invalidate all weight-dependent recurrence/convolution/KV state after synchronization; add long-context cases and the declared checkpoint cache-restore/rebuild policy.

Record each missing provider, worker capture, state mapping, gradient reference and full-model result as an explicit target-project task. The Markdown guide itself is not a verified Hybrid runtime.

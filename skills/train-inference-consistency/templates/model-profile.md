# Model profile — MODEL / REVISION

Guide status: draft. Implementation status in target: unknown. Full-model validation status: not-started.

## Actual identity and architecture

Record checkpoint/config/tensor index, inspected source locations, loaded model classes, tokenizer, precision/quantization and master/deployment relation. Enumerate layer families/schedule, dimensions, expert/router choices, state/cache and multimodal/MTP/speculative paths. Identify variants that must not be conflated and unresolved facts.

## Scope and reuse

State acceptance mode (default consistency-only), requested engines/hardware/topology/batch/context/cache/objective and chosen simple arithmetic path. List covered numerical providers with verified contracts, required glue, genuinely new operators/backward work and unsupported routes. Do not require an external kernel project solely because another model used it.

## Proxy coverage

Record selected original layers/tensors and why all required operator families/transitions/state interactions are represented. Preserve original widths/precision/expert counts and real source weights. List coverage that remains full-model-only.

## Model-specific validation additions

| Stage(s) | Added cases and expected coverage | Capture/reference implementation needed | Current evidence or blocker |
|---|---|---|---|
| C0/C2 | TODO | TODO | not-started |
| C3/C5 | TODO | TODO | not-started |
| C6/C7 | TODO | TODO | not-started |
| C8/C9 | TODO | TODO | not-started |
| C10 | Reproduction and supported correctness scope | TODO | not-started |
| C1/C4 (optional) | Only if performance is requested | TODO when enabled | not-required by default |

## Supported claims

List guide availability separately from implemented target code and measured accepted scopes. For any verified claim, bind exact model/revision, software/hardware, dtype/topology/input ranges, complete evidence and review source. A new Markdown entry alone establishes no runtime support.

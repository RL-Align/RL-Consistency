# MoE routing evidence v1

Qwen3-30B-A3B requires routing evidence in addition to the ordinary v2 gate
envelope and v1 output traces. Implement producers inside real rollout and
training workers. There is no automatic GPU routing collector in this release.

## Envelope and identity

At C3/C5/C8/C9 add `routing_pairs`, with exactly the same case IDs as `pairs`:

```json
{
  "routing_pairs": [
    {"case": "old-logprob", "left": "rollout-old-routes.json", "right": "train-old-routes.json"},
    {"case": "actor-forward", "left": "rollout-actor-routes.json", "right": "train-actor-routes.json"}
  ]
}
```

This is a partial C5 envelope example, not evidence of execution. Paths resolve
under the evidence directory and are snapshotted by the gate loader.

Each ledger has these fields:

| Field | Required value |
|---|---|
| `schema` | `rl-consistency.routing.v1` |
| `producer`, `weights_sha256`, `weight_version`, `inputs_sha256` | Exactly the corresponding output trace's identity |
| `independent` | Boolean `true`, backed by actual independent computation |
| `routing_replay` | Boolean `false`; do not relabel R3 captures |
| `contract_sha256` | Canonical hash of the complete `recipe.routing` object |
| `implementation_sha256` | Actual routing implementation manifest hash |
| `execution_path` | Exact output trace callsite receipt when that trace has one |
| `records` | Nonempty list of logical token/layer records below |

Capture each record after independently selecting routes and observing the
actual dispatch/combine path. Export full rank coverage into logical global IDs;
do not compare raw per-rank expert indices or duplicate replicated records.

## Records

| Field | Meaning |
|---|---|
| `sample`, `position`, `layer` | Stable sample string, nonnegative logical token position, original layer index |
| `experts` | Eight unique integer global expert IDs, 0–127, in original top-k slot order |
| `weights_hex` | 32 bytes encoded as hex: eight little-endian FP32 normalized selected routing weights, in the same slot order, before cast to the expert path's applied dtype |
| `delivered_slots` | Observed delivered slot IDs for this token/layer; each 0–7 exactly once, physical arrival order may vary |
| `returned_experts` | Observed expert IDs after unpermute into logical slot order, equal to `experts` |
| `combine_slots` | Observed accumulation slot order, exactly `[0,1,2,3,4,5,6,7]` in this strict contract |

Do not sort `experts` to hide a slot permutation, cast expert IDs to float, or
recompute/round routing weights while exporting. The weight stage is fixed as
`normalized_pre_cast`; audit the subsequent applied-weight cast and multiplication
in the operator binding and intermediate numeric traces. The ledger does not
replace expert-output and reduction-tree verification.

The validator rejects duplicate keys, missing layers, missing scored tokens,
invalid IDs, nonfinite weights, dropped/duplicated slots and wrong unpermute or
combine receipts. Normalized weights must sum to 1 within `1e-5` as a shape/math
sanity check; **cross-path acceptance still compares exact original bytes**.
Each represented token must cover every proxy layer at C3/C5/C8 or all 48 layers
at C9. Both sides must cover the same token/layer set. Include all fixture-declared
prompt and decode tokens; the validator can check scored-token inclusion but
cannot reconstruct omitted unscored prompt positions from output traces alone.

Equal output traces with different expert IDs, slot order or weight bytes fail
the gate. Receipt errors fail closed. A human must inspect producer coverage,
kernel source, actual process groups and loaded-worker provenance: self-reported
receipt fields cannot establish that a GPU used the declared reduction order.

## Update and resume

At C8 and C9 the ledger's weights/version must match the output capture for each
theta; stale theta0 routes cannot accompany theta1/2 outputs. The regular two
effective updates and master-weight chain checks still apply.

C9's `checkpoint` artifact additionally contains `uninterrupted_routing` and
`resumed_routing`, distinct paths to first-step ledgers paired with its existing
`uninterrupted_trace` and `resumed_trace`. These are compared independently of
the `checkpoint-resume` train/rollout pair. Retaining optimizer, RNG, scheduler
and data-cursor state and matching the resumed update remain mandatory.

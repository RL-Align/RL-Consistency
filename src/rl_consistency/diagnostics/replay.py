"""Execute original callables on one captured input, with reset before every run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..common import publish_json, require
from .capture import DiagnosticWriter
from .locate import Trace, diagnose
from .schema import provenance_identity, raw_tensor, read_tensor, tensor_map, validate_plan


@dataclass
class ReplayEndpoint:
    call: Callable  # named tensor mapping -> named tensor mapping
    reset: Callable  # restores actual weights, caches, RNG, counters before every invocation
    state: Callable  # actual relevant tensor/JSON state after reset, before invocation
    provenance: dict
    execution: dict
    state_complete: bool = False


def replay_boundary(plan, left, right, node, *, inputs, left_endpoint, right_endpoint, output):
    """Two independent calls per endpoint. Results apply only to this replay context.

    Callbacks are supplied by trusted Python integration code, never loaded from
    an evidence JSON. Inputs must reproduce the original left boundary bytes.
    Preserve physical layout, batch composition and engine route in integration
    code; this helper cannot infer or restore opaque engine internals.
    """
    nodes, _ = validate_plan(plan)
    require(node in nodes, "unknown replay boundary")
    a, b = Trace(left, plan), Trace(right, plan)
    require(a.data["capture_id"] != b.data["capture_id"], "replay origins must be independent captures")
    require(a.data["identity"] == b.data["identity"] == plan["identity"], "replay origin identity differs")
    require(node in a.records and node in b.records, "both original boundary observations required")
    inputs = tensor_map(inputs)
    original = a.records[node]["inputs"]
    require(set(inputs) == set(original), "replay input ports differ from original left observation")
    for port, value in inputs.items():
        spec, raw = raw_tensor(value)
        target = original[port]
        require((spec["dtype"], spec["shape"]) == (target["dtype"], target["shape"]) and
                raw == read_tensor(a.root, target), "replay input must retain original left bytes/dtype/shape")
    for endpoint, source in ((left_endpoint, a), (right_endpoint, b)):
        require(provenance_identity(endpoint.provenance) == provenance_identity(source.data["provenance"]),
                "replay provider differs from original capture")
        require(callable(endpoint.call) and callable(endpoint.reset) and callable(endpoint.state), "replay callbacks required")
    local_plan = {**plan, "nodes": [{**nodes[node], "parents": []}]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for side, endpoint in (("left", left_endpoint), ("right", right_endpoint)):
        for repeat in range(2):
            # Clone for isolation, preserving the caller's device/layout/grad mode.
            values = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in inputs.items()}
            endpoint.reset()
            label = side + ("_repeat" if repeat else "")
            writer = DiagnosticWriter(output / label, local_plan, side=side,
                                      provenance=endpoint.provenance, execution=endpoint.execution)
            writer.begin(node, values, state=endpoint.state(), state_complete=endpoint.state_complete)
            writer.end(node, endpoint.call(values))
            paths[label] = writer.finish()
    result = diagnose(local_plan, **paths)
    row = result["rows"][0]
    controlled = (result["repeats_stable_observed"] and row.get("inputs", {}).get("status") == "equal"
                  and row.get("state") == "equal")
    verdict = ("local_replay_mismatch" if controlled and row["status"] == "mismatch" else
               "local_replay_equal" if controlled and row["status"] == "equal" else "inconclusive")
    receipt = {"schema": "diagnostic-replay.v1", "verdict": verdict, "certified": False,
               "origin": {"left": a.source(), "right": b.source(), "node": node}, "comparison": result,
               "limit": "Confirms behavior only under the recorded replay context, not the production root cause."}
    publish_json(output / "replay.json", receipt)
    return receipt

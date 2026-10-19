"""Conservative partial-order localization. No monotonic-error assumption."""
from __future__ import annotations

import math
from pathlib import Path

from ..common import digest, file_hash, is_hash, read_json, require
from .schema import compare_ports, name, provenance_identity, read_tensor, tensor_map, validate_plan


class Trace:
    def __init__(self, path, plan):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.data = read_json(self.path)
        self.sha256 = file_hash(self.path)
        value = self.data
        nodes, _ = validate_plan(plan)
        require(isinstance(value, dict) and value.get("schema") == "diagnostic-trace.v1", "invalid diagnostic trace")
        require(value.get("plan_sha256") == digest(plan), "trace uses a different semantic plan")
        require(name(value.get("capture_id")) and name(value.get("side")), "capture/side identity missing")
        identity = value.get("identity")
        require(isinstance(identity, dict) and name(identity.get("weight_version")) and
                all(is_hash(identity.get(k)) for k in ("weights_sha256", "fixture_sha256", "semantics_sha256")),
                "invalid trace identity")
        provenance, execution = value.get("provenance"), value.get("execution")
        require(isinstance(provenance, dict) and name(provenance.get("provider")) and
                isinstance(provenance.get("manifest"), dict) and provenance["manifest"], "missing worker provenance")
        require(isinstance(execution, dict) and name(execution.get("phase")) and name(execution.get("mode")) and
                isinstance(execution.get("observer_effects"), list) and
                all(name(x) for x in execution["observer_effects"]), "missing execution context")
        records = value.get("records")
        require(isinstance(records, list) and records, "empty trace is not evidence")
        self.records = {}
        for record in records:
            require(isinstance(record, dict) and name(record.get("node")), "invalid boundary record")
            node = record["node"]
            require(node in nodes and node not in self.records, "unknown/duplicate boundary record")
            for group in ("inputs", "outputs"):
                for spec in tensor_map(record.get(group)).values():
                    read_tensor(self.root, spec)
            require(any(math.prod(s["shape"]) > 0 for s in record["outputs"].values()), "empty boundary output")
            state = record.get("state")
            require(isinstance(state, dict) and is_hash(state.get("sha256")) and
                    type(state.get("complete")) is bool and isinstance(state.get("keys"), list) and
                    all(name(k) for k in state["keys"]) and len(set(state["keys"])) == len(state["keys"]),
                    "invalid boundary state fingerprint")
            self.records[node] = record
        missing = value.get("missing")
        require(isinstance(missing, list) and all(name(n) for n in missing) and
                len(set(missing)) == len(missing) and set(missing) == set(nodes) - set(self.records),
                "missing-boundary ledger does not match plan")

    def source(self):
        return {"path": str(self.path), "sha256": self.sha256, "capture_id": self.data["capture_id"],
                "side": self.data["side"], "provenance": self.data["provenance"],
                "execution": self.data["execution"]}


def _repeat(node, original, repeat):
    if repeat is None:
        return "unverified"
    for field in ("side", "identity", "execution"):
        if original.data[field] != repeat.data[field]:
            return "incomparable"
    if provenance_identity(original.data["provenance"]) != provenance_identity(repeat.data["provenance"]):
        return "incomparable"
    if node not in original.records or node not in repeat.records:
        return "missing"
    a, b = original.records[node], repeat.records[node]
    if a["state"] != b["state"] or compare_ports(a["inputs"], b["inputs"], original.root, repeat.root)["status"] != "equal":
        return "incomparable"
    output = compare_ports(a["outputs"], b["outputs"], original.root, repeat.root)["status"]
    return {"equal": "stable_observed", "mismatch": "unstable", "incomparable": "incomparable"}[output]


def diagnose(plan, left, right, *, left_repeat=None, right_repeat=None):
    nodes, order = validate_plan(plan)
    a, b = Trace(left, plan), Trace(right, plan)
    ar = Trace(left_repeat, plan) if left_repeat is not None else None
    br = Trace(right_repeat, plan) if right_repeat is not None else None
    traces = [t for t in (a, b, ar, br) if t is not None]
    ids = [t.data["capture_id"] for t in traces]
    require(len(ids) == len(set(ids)), "each side and repeat requires a distinct capture invocation")
    compatible = a.data["identity"] == b.data["identity"] == plan["identity"]
    rows, by_id, ancestors = [], {}, {}
    for node in order:
        ancestors[node] = set(nodes[node]["parents"])
        for parent in nodes[node]["parents"]:
            ancestors[node].update(ancestors[parent])
        row = {"node": node, "scope": nodes[node]["scope"], "kind": nodes[node]["kind"],
               "parents": nodes[node]["parents"],
               "repeatability": {"left": _repeat(node, a, ar), "right": _repeat(node, b, br)}}
        if not compatible:
            row.update(status="incomparable", reason="weight/fixture/semantic identity differs")
        elif node not in a.records or node not in b.records:
            row.update(status="missing", missing_sides=[label for label, t in (("left", a), ("right", b))
                                                       if node not in t.records])
        else:
            x, y = a.records[node], b.records[node]
            inputs = compare_ports(x["inputs"], y["inputs"], a.root, b.root)
            outputs = compare_ports(x["outputs"], y["outputs"], a.root, b.root)
            state = ("unknown" if not (x["state"]["complete"] and y["state"]["complete"]) else
                     "equal" if x["state"] == y["state"] else "mismatch")
            row.update(status=outputs["status"], inputs=inputs, outputs=outputs, state=state)
            if outputs["status"] == "mismatch":
                row["localization"] = ("same_input_state_candidate" if inputs["status"] == "equal" and state == "equal"
                                       else "input_or_state_changed" if inputs["status"] == "mismatch" or state == "mismatch"
                                       else "unresolved_context")
        rows.append(row)
        by_id[node] = row
    frontier = []
    for row in rows:
        upstream = ancestors[row["node"]]
        row["unresolved_ancestors"] = [n for n in order if n in upstream and by_id[n]["status"] in {"missing", "incomparable"}]
        if row["status"] == "mismatch" and not any(by_id[p]["status"] == "mismatch" for p in upstream):
            frontier.append(row["node"])
            row["frontier"] = "upstream_unresolved" if row["unresolved_ancestors"] else "earliest_observed"
    all_equal = all(r["status"] == "equal" for r in rows)
    stable = all(v == "stable_observed" for r in rows for v in r["repeatability"].values())
    verdict = ("mismatch" if any(r["status"] == "mismatch" for r in rows) else
               "equal_observed" if all_equal and stable else "inconclusive")
    return {"schema": "diagnostic-result.v1", "plan_sha256": digest(plan), "verdict": verdict,
            "certified": False, "all_observed_outputs_equal": all_equal, "repeats_stable_observed": stable,
            "frontier": frontier, "rows": rows,
            "coverage": {"expected": len(nodes), "left": len(a.records), "right": len(b.records),
                         "opaque": [n for n in order if nodes[n]["kind"] == "opaque"]},
            "sources": {label: trace.source() for label, trace in
                        (("left", a), ("right", b), ("left_repeat", ar), ("right_repeat", br)) if trace},
            "limits": ["First observed boundary in the declared semantic DAG; not proof of the first internal cause.",
                       "Mappings, state completeness and provenance are producer declarations, not authenticated attestations.",
                       "Instrumentation may change execution. Confirm on the original execution route after diagnosis."]}

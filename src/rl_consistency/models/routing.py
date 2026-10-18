"""Validate independent MoE routing ledgers. Integer expert IDs stay integers.

These checks reject incomplete/inconsistent captures; they do not attest that a
producer actually instrumented its kernels. Review the loaded worker bindings.
"""
from __future__ import annotations

import math
import struct
from ..common import digest, is_hash, require

SCHEMA = "rl-consistency.routing.v1"


def validate_routing(value, output_trace, recipe, layers):
    require(isinstance(value, dict) and value.get("schema") == SCHEMA, "routing ledger v1 required")
    require(value.get("independent") is True and value.get("routing_replay") is False,
            "routing must be independently computed; replay is diagnostic only")
    require(value.get("contract_sha256") == digest(recipe["routing"]), "routing contract mismatch")
    require(is_hash(value.get("implementation_sha256")), "router implementation fingerprint missing")
    for key in ("producer", "weights_sha256", "weight_version", "inputs_sha256"):
        require(value.get(key) == output_trace[key], f"routing/output {key} mismatch")
    if "execution_path" in output_trace:
        require(value.get("execution_path") == output_trace["execution_path"], "routing/output callsite mismatch")
    records = value.get("records")
    require(isinstance(records, list) and records, "nonempty routing records required")
    indexed, tokens = {}, set()
    contract = recipe["routing"]
    slots = list(range(contract["top_k"]))
    for record in records:
        require(isinstance(record, dict), "routing record must be an object")
        sample, position, layer = record.get("sample"), record.get("position"), record.get("layer")
        require(isinstance(sample, str) and sample and type(position) is int and position >= 0
                and type(layer) is int and layer in layers, "invalid routing logical coordinate")
        key = (sample, position, layer)
        require(key not in indexed, "duplicate routing coordinate")
        experts = record.get("experts")
        require(isinstance(experts, list) and len(experts) == len(slots)
                and all(type(e) is int and 0 <= e < contract["num_experts"] for e in experts)
                and len(set(experts)) == len(experts), "invalid top-k expert IDs")
        delivered = record.get("delivered_slots")
        require(isinstance(delivered, list) and all(type(s) is int for s in delivered)
                and sorted(delivered) == slots, "dropped or duplicated expert dispatch slot")
        returned = record.get("returned_experts")
        require(isinstance(returned, list) and all(type(e) is int for e in returned)
                and returned == experts, "dispatch/unpermute returned the wrong expert")
        combine = record.get("combine_slots")
        require(isinstance(combine, list) and all(type(s) is int for s in combine)
                and combine == slots, "expert combine order violates strict contract")
        encoded = record.get("weights_hex")
        require(isinstance(encoded, str), "normalized routing weights must be original float32 bytes")
        try:
            raw = bytes.fromhex(encoded)
        except ValueError:
            require(False, "invalid routing weight hex")
        require(len(raw) == 4 * len(slots), "routing weight shape/dtype mismatch")
        weights = struct.unpack("<" + "f" * len(slots), raw)
        require(all(math.isfinite(w) and 0 <= w <= 1 for w in weights), "invalid routing probability")
        require(abs(sum(weights) - 1.0) <= 1e-5, "top-k weights are not normalized")
        # Normalization sanity tolerance above does not relax the byte comparison.
        indexed[key] = (tuple(experts), raw)
        tokens.add((sample, position))
    require(set(indexed) == {(sample, pos, layer) for sample, pos in tokens for layer in layers},
            "routing capture misses layers for one or more tokens")
    output_tokens = {(r["sample"], r["position"]) for r in output_trace["records"]}
    require(output_tokens <= tokens, "routing capture misses scored tokens")
    return indexed


def compare_routing(left, right, left_output, right_output, recipe, layers):
    a = validate_routing(left, left_output, recipe, layers)
    b = validate_routing(right, right_output, recipe, layers)
    require(set(a) == set(b), "routing logical token/layer coverage differs")
    mismatches = [key for key in sorted(a) if a[key] != b[key]]
    return {"passed": not mismatches, "token_layers": len(a), "mismatches": len(mismatches),
            "first_mismatch": list(mismatches[0]) if mismatches else None}


def check_routing_evidence(gate, evidence, recipe, artifact):
    items = evidence.get("routing_pairs", [])
    require(isinstance(items, list), "routing_pairs must be a list")
    ids = [item["case"] for item in items]
    require(len(ids) == len(set(ids)) and set(ids) == set(recipe["coverage"][gate]),
            f"{gate} requires independent routing pairs for every output case")
    outputs = {pair["case"]: pair for pair in evidence["pairs"]}
    layers = list(range(recipe["model"]["full_layers"])) if gate == "C9" else recipe["model"]["proxy_source_layers"]
    results = []
    for item in items:
        require(item["left"] != item["right"], "cannot compare a routing artifact with itself")
        output = outputs[item["case"]]
        result = compare_routing(artifact(item["left"]), artifact(item["right"]),
                                 artifact(output["left"]), artifact(output["right"]), recipe, layers)
        results.append({"case": item["case"], **result})
    if gate == "C9":
        checkpoint = artifact(evidence["checkpoint"])
        require(checkpoint["uninterrupted_routing"] != checkpoint["resumed_routing"], "resume requires separate routing captures")
        result = compare_routing(artifact(checkpoint["uninterrupted_routing"]), artifact(checkpoint["resumed_routing"]),
                                 artifact(checkpoint["uninterrupted_trace"]), artifact(checkpoint["resumed_trace"]), recipe, layers)
        results.append({"case": "uninterrupted-vs-resumed-routing", **result})
    return results

"""Explicit semantic DAG and raw tensor storage. No model-name registry."""
from __future__ import annotations

import hashlib
import copy
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

from ..common import contained, digest, is_hash, require

DTYPES = {"float16": (2, "e"), "bfloat16": (2, None), "float32": (4, "f"),
          "float64": (8, "d"), "int64": (8, "q"), "int32": (4, "i"),
          "int16": (2, "h"), "int8": (1, "b"), "uint8": (1, "B"), "bool": (1, "?")}


def name(value):
    return isinstance(value, str) and bool(value.strip())


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") == "diagnostic-plan.v1", "invalid diagnostic plan")
    identity = plan.get("identity")
    require(isinstance(identity, dict), "plan identity required")
    for field in ("weights_sha256", "fixture_sha256", "semantics_sha256"):
        require(is_hash(identity.get(field)), f"identity.{field} must be SHA256")
    require(name(identity.get("weight_version")), "weight version required")
    nodes = plan.get("nodes")
    require(isinstance(nodes, list) and nodes, "plan must declare expected boundaries")
    by_id, scopes = {}, set()
    for node in nodes:
        require(isinstance(node, dict) and name(node.get("id")), "invalid node")
        key = node["id"]
        require(key not in by_id, f"duplicate node: {key}")
        parents = node.get("parents")
        require(isinstance(parents, list) and all(name(p) for p in parents), "parents must be node IDs")
        require(len(parents) == len(set(parents)), "duplicate parent")
        require(node.get("kind") in {"module", "operator", "opaque"}, "invalid boundary kind")
        scope = node.get("scope")
        require(isinstance(scope, dict), "explicit logical scope required")
        for field in ("sample", "layer", "boundary", "shard"):
            require(name(scope.get(field)), f"scope.{field} required")
        positions = scope.get("positions")
        require(isinstance(positions, list) and positions and
                all(type(p) is int and p >= 0 for p in positions) and len(set(positions)) == len(positions),
                "scope.positions must be unique logical positions")
        require(type(scope.get("call")) is int and scope["call"] >= 0, "scope.call required")
        scope_hash = digest(scope)
        require(scope_hash not in scopes, "duplicate logical scope; use distinct invocation/partition IDs")
        scopes.add(scope_hash)
        by_id[key] = node
    require(all(p in by_id for n in nodes for p in n["parents"]), "unknown dependency")
    order, remaining, seen = [], list(by_id), set()
    while remaining:
        ready = [n for n in remaining if set(by_id[n]["parents"]) <= seen]
        require(ready, "diagnostic dependency graph contains a cycle")
        order.extend(ready)
        seen.update(ready)
        remaining = [n for n in remaining if n not in seen]
    return by_id, order


def raw_tensor(tensor):
    """Detach a snapshot, never modify the original tensor or autograd graph."""
    import torch
    require(isinstance(tensor, torch.Tensor), "capture ports must contain tensors")
    require(sys.byteorder == "little", "capture currently requires a little-endian host")
    dtype = str(tensor.dtype).removeprefix("torch.")
    require(dtype in DTYPES and tensor.layout == torch.strided, "unsupported tensor dtype/layout")
    snapshot = tensor.detach().contiguous().cpu()
    raw = bytes(snapshot.reshape(-1).view(torch.uint8).tolist())
    spec = {"dtype": dtype, "shape": list(snapshot.shape), "sha256": hashlib.sha256(raw).hexdigest()}
    validate_bytes(spec, raw)
    return spec, raw


def validate_bytes(spec, raw):
    require(isinstance(spec, dict) and spec.get("dtype") in DTYPES, "invalid tensor dtype")
    shape = spec.get("shape")
    require(isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape), "invalid tensor shape")
    size, fmt = DTYPES[spec["dtype"]]
    require(len(raw) == math.prod(shape) * size, "tensor byte length does not match shape")
    require(is_hash(spec.get("sha256")) and hashlib.sha256(raw).hexdigest() == spec["sha256"], "tensor hash mismatch")
    if spec["dtype"] == "bfloat16":
        require(all((value[0] & 0x7f80) != 0x7f80 for value in struct.iter_unpack("<H", raw)), "non-finite tensor")
    elif spec["dtype"].startswith("float"):
        require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<" + fmt, raw)), "non-finite tensor")
    elif spec["dtype"] == "bool":
        require(all(v in (0, 1) for v in raw), "noncanonical bool payload")


def read_tensor(root, spec):
    require(isinstance(spec, dict) and name(spec.get("path")), "tensor blob path required")
    raw = contained(root, spec["path"]).read_bytes()
    validate_bytes(spec, raw)
    return raw


def tensor_map(value, *, allow_empty=False):
    require(isinstance(value, dict) and (value or allow_empty) and all(name(k) for k in value),
            "ports must be a named tensor mapping")
    return value


def snapshot_state(state, *, complete):
    """Hash actual tensor state plus JSON metadata, not a caller's claimed digest."""
    require(isinstance(state, dict) and all(name(k) for k in state), "state must be a named mapping")
    require(type(complete) is bool, "state completeness must be explicit")
    records = {}
    for key, value in state.items():
        # Do not import torch until a tensor-like object is actually supplied.
        if hasattr(value, "detach"):
            spec, _ = raw_tensor(value)
            records[key] = {"tensor": spec}
        else:
            records[key] = {"json": value}
    return {"sha256": digest(records), "complete": complete, "keys": sorted(records)}


def provenance_identity(provenance):
    """Allow a fresh process ID, retaining actual code, host, rank and routes."""
    result = copy.deepcopy(provenance)
    manifest = result.get("manifest", {})
    if manifest.get("schema") == "rl-consistency.worker.v1":
        manifest.pop("pid", None)
    return result


class TensorStore:
    def __init__(self, root, max_bytes):
        require(type(max_bytes) is int and max_bytes > 0, "positive capture budget required")
        self.root, self.max_bytes, self.used = Path(root), max_bytes, 0
        (self.root / "tensors").mkdir()

    def put(self, tensor):
        require(hasattr(tensor, "numel") and hasattr(tensor, "element_size"), "capture requires a tensor")
        require(self.used + tensor.numel() * tensor.element_size() <= self.max_bytes,
                "capture budget exceeded; narrow the boundary plan")
        spec, raw = raw_tensor(tensor)
        require(self.used + len(raw) <= self.max_bytes, "capture budget exceeded; narrow the boundary plan")
        self.used += len(raw)
        relative = f"tensors/{spec['sha256']}.bin"
        target = self.root / relative
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tensor-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                os.link(temporary, target)
            finally:
                os.unlink(temporary)
        else:
            require(target.read_bytes() == raw, "existing tensor blob differs")
        return {**spec, "path": relative}


def compare_ports(left, right, left_root, right_root):
    tensor_map(left); tensor_map(right)
    if set(left) != set(right):
        return {"status": "incomparable", "reason": "port coverage differs"}
    differences, incompatible = [], []
    for port in sorted(left):
        a, b = left[port], right[port]
        raw_a, raw_b = read_tensor(left_root, a), read_tensor(right_root, b)
        if (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
            incompatible.append(port)
        elif raw_a != raw_b:
            offset = next(i for i, (x, y) in enumerate(zip(raw_a, raw_b)) if x != y)
            differences.append({"port": port, "first_byte": offset,
                                "flat_element": offset // DTYPES[a["dtype"]][0],
                                "left_sha256": a["sha256"], "right_sha256": b["sha256"]})
    # A partial mapping must not be presented as a complete aligned comparison.
    if incompatible:
        return {"status": "incomparable", "reason": "dtype/shape differs", "ports": incompatible,
                "aligned_port_differences": differences}
    return {"status": "mismatch" if differences else "equal", "differences": differences}

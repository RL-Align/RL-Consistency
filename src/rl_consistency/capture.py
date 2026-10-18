"""Explicit worker-side capture. No imports, hooks, GPU jobs or dtype casts on import."""
from __future__ import annotations

import copy
import os
import re
import socket
import sys
from pathlib import Path
from .common import digest, file_hash, is_hash, positive_int, publish_json, require
from .compare import DTYPES, tensor_record, validate_records, validate_trace


def worker_manifest(*, role, rank, world_size, module_names, mapped_libraries=True):
    """Fingerprint selected already-imported modules and Linux mapped shared libraries.

    Call inside each relevant worker after operators have executed. File bytes
    are read from disk; this is not an attestation of process memory or JIT code.
    """
    require(role in {"training", "rollout", "reference", "inference"}, "invalid worker role")
    require(positive_int(world_size) and type(rank) is int and 0 <= rank < world_size,
            "invalid worker rank/world size")
    require(isinstance(module_names, (list, tuple)) and module_names, "select loaded modules explicitly")
    require(len(set(module_names)) == len(module_names), "duplicate module names")
    artifacts = []
    for name in sorted(module_names):
        module = sys.modules.get(name)
        require(module is not None and getattr(module, "__file__", None), f"module not loaded from a file: {name}")
        path = Path(module.__file__).resolve(strict=True)
        artifacts.append({"worker": role, "rank": rank, "kind": "python_module", "module": name,
                          "path": str(path), "sha256": file_hash(path)})
    if mapped_libraries:
        maps = Path("/proc/self/maps")
        require(maps.is_file(), "mapped library capture requires Linux /proc/self/maps")
        paths = set()
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            value = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[5])
            if ".so" in Path(value).name:
                require(not value.endswith(" (deleted)"), "mapped library was deleted; cannot fingerprint it")
                paths.add(Path(value).resolve(strict=True))
        require(paths, "no mapped shared libraries found")
        for path in sorted(paths):
            artifacts.append({"worker": role, "rank": rank, "kind": "mapped_library",
                              "path": str(path), "sha256": file_hash(path)})
    return {"schema": "rl-consistency.worker.v1", "role": role, "rank": rank,
            "world_size": world_size, "hostname": socket.gethostname(), "pid": os.getpid(),
            "mapped_libraries_collected": bool(mapped_libraries), "loaded_artifacts": artifacts}


def token_records(sample, positions, node, tensor, *, active_mask=None):
    """Capture [token, ...] tensors using explicit global target-token positions.

    Padding removal, causal shift and distributed gathering belong to the caller.
    A bool mask selects rows without converting the floating-point dtype.
    """
    import torch
    require(isinstance(tensor, torch.Tensor) and tensor.ndim >= 1, "expected [token, ...] tensor")
    require(isinstance(positions, (list, tuple)) and len(positions) == tensor.shape[0],
            "one explicit position is required per token row")
    require(all(type(p) is int and p >= 0 for p in positions) and len(set(positions)) == len(positions),
            "token positions must be unique non-negative integers")
    dtype = str(tensor.dtype).removeprefix("torch.")
    require(dtype in DTYPES, "capture supports floating-point outputs only")
    if active_mask is None:
        indices = list(range(len(positions)))
    else:
        require(isinstance(active_mask, torch.Tensor) and active_mask.dtype == torch.bool
                and tuple(active_mask.shape) == (len(positions),), "active_mask must be a bool vector")
        indices = active_mask.detach().cpu().nonzero().flatten().tolist()
    # One device-to-host transfer per call, not per token. Detach leaves the live
    # computation and its gradients unchanged; capture synchronization affects timing.
    value = tensor.detach().cpu()
    records = [tensor_record(sample, positions[i], node, value[i]) for i in indices]
    if records:
        validate_records(records, dtype)
    return dtype, records


class TraceWriter:
    """Accumulate one independently computed trace; publish only exact coverage.

    A writer belongs to one process, weight version and homogeneous output dtype.
    expected_keys comes from the reviewed fixture, not the observed tensor length.
    """
    def __init__(self, *, producer, weight_version, weights_sha256, inputs_sha256,
                 semantics_sha256, implementation_manifest, independent, expected_keys,
                 reference_kind=None, execution_path=None):
        manifest = copy.deepcopy(implementation_manifest)
        require(isinstance(manifest, dict) and manifest.get("schema") == "rl-consistency.worker.v1",
                "worker implementation manifest required")
        require(manifest.get("role") == producer, "worker/producer role mismatch")
        require(manifest.get("loaded_artifacts") and all(is_hash(x.get("sha256"))
                for x in manifest["loaded_artifacts"]), "worker artifact hashes missing")
        keys = [tuple(key) for key in expected_keys]
        require(keys and len(set(keys)) == len(keys), "expected coverage must be nonempty and unique")
        for key in keys:
            require(len(key) == 3 and isinstance(key[0], str) and key[0]
                    and type(key[1]) is int and key[1] >= 0 and isinstance(key[2], str) and key[2],
                    "invalid expected logical key")
        require(independent is True, "capture must independently compute its outputs")
        self.expected = set(keys)
        self.records = {}
        self.dtype = None
        implementations = sorted(
            [{"kind": item.get("kind"), "module": item.get("module"), "sha256": item["sha256"]}
             for item in manifest["loaded_artifacts"]], key=digest)
        self.metadata = {"schema": "rl-consistency.trace.v1", "producer": producer,
                         "weight_version": weight_version, "weights_sha256": weights_sha256,
                         "inputs_sha256": inputs_sha256, "semantics_sha256": semantics_sha256,
                         "implementation_sha256": digest(implementations), "independent": independent,
                         "capture": {"worker_manifest": manifest,
                                     "expected_keys_sha256": digest(sorted(keys))}}
        if reference_kind is not None:
            self.metadata["reference_kind"] = reference_kind
        if execution_path is not None:
            self.metadata["execution_path"] = copy.deepcopy(execution_path)
        # Validate identities before accepting data (the probe is never saved).
        validate_trace({**self.metadata, "dtype": "float32", "records": [
            {"sample": keys[0][0], "position": keys[0][1], "node": keys[0][2], "shape": [], "hex": "00000000"}]})
        self.closed = False

    def add_records(self, dtype, records):
        require(not self.closed, "trace already published")
        rows = copy.deepcopy(records)
        keys = validate_records(rows, dtype).keys()
        require(self.dtype in (None, dtype), "mixed trace dtypes are not allowed")
        require(set(keys) <= self.expected, "captured an undeclared logical token/node")
        require(not (set(keys) & self.records.keys()), "duplicate logical tensor across capture calls")
        self.records.update(zip(keys, rows))
        self.dtype = dtype

    def add_tokens(self, sample, positions, node, tensor, *, active_mask=None):
        require(not self.closed, "trace already published")
        dtype, rows = token_records(sample, positions, node, tensor, active_mask=active_mask)
        if rows:
            self.add_records(dtype, rows)

    def finish(self, output):
        require(not self.closed, "trace already published")
        require(self.records.keys() == self.expected, "capture coverage incomplete")
        trace = {**self.metadata, "dtype": self.dtype,
                 "records": [self.records[key] for key in sorted(self.records)]}
        validate_trace(trace)
        publish_json(output, trace)
        self.closed = True
        return {"output": str(output), "sha256": file_hash(output), "records": len(self.records)}

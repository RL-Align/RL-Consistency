"""Compare raw little-endian tensor bytes after logical token alignment."""
from __future__ import annotations

import math
import struct
import sys
from .common import require, is_hash

DTYPES = {"float16": (2, "e"), "bfloat16": (2, None), "float32": (4, "f"), "float64": (8, "d")}


def unpack(dtype, data):
    size, code = DTYPES[dtype]
    if dtype == "bfloat16":
        return [struct.unpack("<f", b"\0\0" + data[i:i+2])[0] for i in range(0, len(data), 2)]
    return list(struct.unpack("<" + code * (len(data) // size), data))


def validate_trace(trace):
    require(trace.get("schema") == "rl-consistency.trace.v1", "unsupported trace schema")
    require(trace.get("producer") in {"rollout", "training", "reference", "inference"}, "trace producer missing")
    for key in ("weights_sha256", "inputs_sha256", "semantics_sha256", "implementation_sha256"):
        require(is_hash(trace.get(key)), f"trace {key} missing")
    require(isinstance(trace.get("weight_version"), str) and bool(trace["weight_version"]), "weight version missing")
    require(trace.get("independent") is True, "trace must be independently computed")
    return validate_records(trace.get("records"), trace.get("dtype"))


def validate_records(records, dtype):
    """Validate bytes without inventing model/execution identities."""
    require(dtype in DTYPES, "unsupported trace dtype")
    require(isinstance(records, list) and records, "empty traces cannot pass")
    out = {}
    for record in records:
        sample, position, node = record.get("sample"), record.get("position"), record.get("node")
        require(isinstance(sample, str) and sample and type(position) is int and position >= 0
                and isinstance(node, str) and node, "invalid logical tensor identity")
        key = (sample, position, node)
        require(key not in out, f"duplicate logical tensor: {key}")
        shape = record.get("shape")
        require(isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape), "invalid shape")
        try:
            data = bytes.fromhex(record["hex"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("trace requires hexadecimal tensor bytes") from exc
        require(len(data) == math.prod(shape) * DTYPES[dtype][0], "shape/byte count mismatch")
        values = unpack(dtype, data)
        require(all(math.isfinite(x) for x in values), f"non-finite tensor: {key}")
        out[key] = (shape, data, values)
    return out


def compare_traces(left, right, *, accuracy=False, atol=0.0, rtol=0.0):
    a, b = validate_trace(left), validate_trace(right)
    for key in ("weight_version", "weights_sha256", "inputs_sha256", "semantics_sha256"):
        require(left[key] == right[key], f"comparison identity mismatch: {key}")
    return _compare(a, b, left["dtype"], right["dtype"], accuracy=accuracy, atol=atol, rtol=rtol)


def compare_records(left, right, left_dtype, right_dtype, *, accuracy=False, atol=0.0, rtol=0.0):
    """Numerical diagnostic only; callers must separately check provenance."""
    a, b = validate_records(left, left_dtype), validate_records(right, right_dtype)
    return _compare(a, b, left_dtype, right_dtype, accuracy=accuracy, atol=atol, rtol=rtol)


def _compare(a, b, left_dtype, right_dtype, *, accuracy, atol, rtol):
    if not accuracy:
        require(left_dtype == right_dtype, "bitwise comparison requires equal dtype")
    require(a.keys() == b.keys(), "logical tensor coverage mismatch")
    require(all(math.isfinite(x) and x >= 0 for x in (atol, rtol)), "invalid tolerance")
    mismatch_elements = 0
    first = None
    max_abs = 0.0
    total = 0
    for key in sorted(a):
        shape, raw_a, values_a = a[key]
        shape_b, raw_b, values_b = b[key]
        require(shape == shape_b, f"shape mismatch at {key}")
        size = DTYPES[left_dtype][0]
        for index, (x, y) in enumerate(zip(values_a, values_b)):
            total += 1
            delta = abs(x - y)
            max_abs = max(max_abs, delta)
            same = delta <= atol + rtol * abs(y) if accuracy else (
                raw_a[index*size:(index+1)*size] == raw_b[index*size:(index+1)*size])
            if not same:
                mismatch_elements += 1
                if first is None:
                    first = {"sample": key[0], "position": key[1], "node": key[2], "element": index}
    return {"passed": mismatch_elements == 0, "comparison": "accuracy" if accuracy else "bitwise",
            "elements": total, "mismatch_elements": mismatch_elements,
            "max_abs": max_abs if math.isfinite(max_abs) else None,
            "first_mismatch": first}


def tensor_record(sample, position, node, tensor):
    """Optional PyTorch capture helper. Call inside the actual engine worker.

    Trace metadata and logical unpadding are the producer's responsibility.
    No GPU or torch dependency is needed for reading/comparing the result.
    """
    import torch
    require(sys.byteorder == "little", "capture currently requires a little-endian host")
    dtype = str(tensor.dtype).removeprefix("torch.")
    require(dtype in DTYPES, "capture supports floating-point outputs only")
    value = tensor.detach().contiguous().cpu()
    raw = bytes(value.reshape(-1).view(torch.uint8).tolist())
    return {"sample": sample, "position": position, "node": node,
            "shape": list(value.shape), "hex": raw.hex()}

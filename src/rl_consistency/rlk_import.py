"""Read the pinned RL-Kernel Vime mismatch sidecars as diagnostics only."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from .adapters import RLK_COMMIT
from .common import ContractError, contained, digest, file_hash, positive_int, publish_json, require
from .compare import DTYPES, compare_records

SIDECAR_SCHEMA = "rlkernel.vime_rocm_attention_mismatch_sidecar.v1"


def response_positions(total, response, cp, cp_rank):
    """Response-relative target indices owned by Vime's two-ended CP layout.

    Ownership is by the preceding logit position. CP1 is the full response.
    This is the pinned non-allgather, EP=PP=1 layout, not a general CP adapter.
    """
    require(positive_int(total) and positive_int(response) and total > response,
            "positive prompt and response lengths required")
    require(positive_int(cp) and type(cp_rank) is int and 0 <= cp_rank < cp, "invalid CP rank/size")
    if cp == 1:
        return list(range(response))
    width = (total + 2 * cp - 1) // (2 * cp)
    first_logit = total - response - 1
    owned = []
    for block in (cp_rank, 2 * cp - 1 - cp_rank):
        lo = max(first_logit, block * width)
        hi = min(total - 1, (block + 1) * width)
        owned.extend(range(lo - first_logit, hi - first_logit))
    return owned


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ContractError("sidecar import needs PyTorch >=2.6 in this interpreter; install it explicitly") from exc
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    require(version and tuple(map(int, version.groups())) >= (2, 6), "sidecar import needs PyTorch >=2.6")
    require(sys.byteorder == "little", "sidecar import requires a little-endian host")
    return torch


def _vector(value, label, torch):
    require(isinstance(value, torch.Tensor) and value.ndim == 1 and value.layout == torch.strided,
            f"{label} must be a dense 1-D tensor from the pinned hook")
    return value.detach().cpu().contiguous()


def _raw(value, torch):
    return bytes(value.view(torch.uint8).tolist())


def _payload(payload, tp, cp, world_size, torch):
    require(isinstance(payload, dict) and payload.get("schema_version") == SIDECAR_SCHEMA,
            "unsupported RL-Kernel sidecar schema")
    require(type(payload.get("tensor_parallel_size")) is int and payload["tensor_parallel_size"] == tp
            and type(payload.get("context_parallel_size")) is int and payload["context_parallel_size"] == cp,
            "sidecar topology differs from explicit import topology")
    rank, call = payload.get("rank"), payload.get("call_index")
    require(type(rank) is int and 0 <= rank < world_size and type(call) is int and call >= 0,
            "invalid sidecar rank/call index")
    fields = ("train_log_probs", "rollout_log_probs", "loss_masks", "total_lengths", "response_lengths")
    require(all(isinstance(payload.get(key), (list, tuple)) for key in fields), "missing sidecar lists")
    count = len(payload[fields[0]])
    require(count > 0 and all(len(payload[key]) == count for key in fields), "sidecar sample list lengths differ or are empty")
    samples = []
    for train, rollout, mask, total, response in zip(*(payload[key] for key in fields)):
        positions = response_positions(total, response, cp, (rank // tp) % cp)
        train, rollout = _vector(train, "training", torch), _vector(rollout, "rollout", torch)
        mask = _vector(mask, "loss mask", torch)
        dtype = str(train.dtype).removeprefix("torch.")
        require(dtype in DTYPES and train.dtype == rollout.dtype, "sidecar dtype mismatch or unsupported dtype")
        require(train.numel() == rollout.numel() == len(positions), "CP shard length does not match logit ownership")
        require(mask.numel() == response and bool(((mask == 0) | (mask == 1)).all()),
                "loss mask must contain exactly one binary entry per response token")
        masks = mask.bool().tolist()
        samples.append({"dtype": dtype, "positions": positions, "total": total, "response": response,
                        "mask": masks, "training": _raw(train, torch), "rollout": _raw(rollout, torch)})
    return rank, call, samples


def inspect_sidecars(source, *, tp, cp, world_size, layout):
    """Validate rank/call/replica coverage and compare retained raw bytes.

    A complete missing call or sample cannot be detected without an external
    fixture. These artifacts lack model, stable sample and weight identities.
    """
    require(layout == "vime-zigzag", "only explicit vime-zigzag layout is supported; allgather is unsupported")
    require(all(positive_int(x) for x in (tp, cp, world_size)) and world_size % (tp * cp) == 0,
            "world size must be divisible by TP*CP (PP=EP=1)")
    source = Path(source).resolve(strict=True)
    require(source.is_dir(), "source must be the mismatch-sidecar directory")
    paths = sorted(source.glob("*.pt"))
    require(paths, "no .pt mismatch sidecars found")
    torch = _torch()
    entries, artifacts = {}, []
    for candidate in paths:
        path = contained(source, candidate.name)
        before = file_hash(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ContractError(f"cannot read {candidate.name} with weights_only=True: {type(exc).__name__}") from exc
        rank, call, samples = _payload(payload, tp, cp, world_size, torch)
        require(file_hash(path) == before, "sidecar changed while being read")
        require((rank, call) not in entries, "duplicate rank/call sidecar")
        entries[rank, call] = samples
        artifacts.append({"path": candidate.name, "sha256": before, "rank": rank, "call_index": call})
    calls = sorted({call for _, call in entries})
    require(set(entries) == {(rank, call) for rank in range(world_size) for call in calls},
            "missing rank or call sidecars")
    require(calls == list(range(calls[-1] + 1)), "sidecar calls must be contiguous from zero")
    training, rollout, dtypes = [], [], set()
    sample_count = 0
    for dp in range(world_size // (tp * cp)):
        for call in calls:
            base = dp * tp * cp
            count = len(entries[base, call])
            require(all(len(entries[base + r, call]) == count for r in range(tp * cp)),
                    "sample count differs across model-parallel ranks")
            for cp_rank in range(cp):
                leader = base + cp_rank * tp
                for replica in range(1, tp):
                    require(entries[leader, call] == entries[leader + replica, call],
                            "TP replicas differ in tensor bytes, mask or lengths")
            for index in range(count):
                sample_count += 1
                exemplar = entries[base, call][index]
                seen = set()
                for cp_rank in range(cp):
                    sample = entries[base + cp_rank * tp, call][index]
                    require(all(sample[key] == exemplar[key] for key in ("dtype", "total", "response", "mask")),
                            "sample identity/mask differs across CP shards")
                    dtypes.add(sample["dtype"])
                    size = DTYPES[sample["dtype"]][0]
                    for local, position in enumerate(sample["positions"]):
                        require(position not in seen, "overlapping CP target positions")
                        seen.add(position)
                        if not sample["mask"][position]:
                            continue
                        identity = {"sample": f"dp:{dp}/call:{call}/sample:{index}",
                                    "position": sample["total"] - sample["response"] + position,
                                    "node": "selected_logprob", "shape": []}
                        training.append({**identity, "hex": sample["training"][local*size:(local+1)*size].hex()})
                        rollout.append({**identity, "hex": sample["rollout"][local*size:(local+1)*size].hex()})
                require(seen == set(range(exemplar["response"])), "incomplete CP response coverage")
    require(len(dtypes) == 1, "mixed sidecar dtypes are unsupported")
    dtype = next(iter(dtypes))
    result = compare_records(rollout, training, dtype, dtype)
    return {"schema": "rl-consistency.rlk-diagnostic.v1", "certified": False,
            "gate_eligible": False, "source_format_commit": RLK_COMMIT,
            "source_artifacts": artifacts, "source_manifest_sha256": digest(artifacts),
            "topology": {"tp": tp, "cp": cp, "world_size": world_size, "pp": 1, "ep": 1, "layout": layout},
            "calls": calls, "sample_count": sample_count, "comparison": result,
            "dtype": dtype, "rollout_records": rollout, "training_records": training,
            "missing_identities": ["complete_loaded_weights", "weight_version_per_call", "fixture_and_stable_sample_ids",
                                   "scoring_contract", "worker_loaded_implementations", "independent_recompute_provenance"],
            "scope": "active selected logprobs in supplied sidecars; call indices are not optimizer steps"}


def import_sidecars(source, output, *, tp, cp, world_size, layout):
    report = inspect_sidecars(source, tp=tp, cp=cp, world_size=world_size, layout=layout)
    publish_json(output, report)
    return {"output": str(output), "sha256": file_hash(output), "certified": False, "gate_eligible": False,
            "sample_count": report["sample_count"], "comparison": report["comparison"]}

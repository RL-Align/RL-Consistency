"""Synthetic CPU fault injection, not evidence for any supported LLM."""
from pathlib import Path

from ..capture import worker_manifest
from ..common import ContractError, digest, publish_json
from .capture import DiagnosticWriter, ModuleCapture, discover_modules
from .replay import ReplayEndpoint, replay_boundary
from .report import run_matrix
from .schema import raw_tensor


def run_demo(output):
    try:
        import torch
    except ImportError as exc:
        raise ContractError("diagnostic-demo needs optional PyTorch; diagnose itself uses the standard library") from exc

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x * 1

    class InjectedFault(torch.nn.Module):
        def forward(self, x):
            return x + 0.125

    class RoundDown(torch.nn.Module):
        def forward(self, x):
            return torch.floor(x)

    left = torch.nn.Sequential(Identity(), Identity(), RoundDown())
    right = torch.nn.Sequential(Identity(), InjectedFault(), RoundDown())
    inputs = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    fixture, _ = raw_tensor(inputs)
    ids = ["embedding", "block.10.native", "block.2.output"]
    nodes = [{"id": key, "parents": [] if i == 0 else [ids[i-1]], "kind": "module",
              "scope": {"sample": "synthetic-only", "positions": [0, 1], "layer": str(i),
                        "boundary": "output", "call": 0, "shard": "full"}} for i, key in enumerate(ids)]
    plan = {"schema": "diagnostic-plan.v1", "identity": {"weights_sha256": digest({"parameters": []}),
            "weight_version": "synthetic-no-weights", "fixture_sha256": digest(fixture),
            "semantics_sha256": digest({"expected": "floor(identity(identity(x)))"})}, "nodes": nodes}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    publish_json(root / "plan.json", plan)
    publish_json(root / "inventory.json", discover_modules(left))
    execution = {"phase": "synthetic-forward", "mode": "cpu-eager", "observer_effects": []}
    manifest = worker_manifest(role="reference", rank=0, world_size=1,
                               module_names=[__name__, "torch"], mapped_libraries=False)
    provenance = {side: {"provider": "synthetic-" + side, "manifest": manifest} for side in ("left", "right")}
    paths = {}
    for side, model in (("left", left), ("right", right)):
        for repeat in range(2):
            label = side + ("_repeat" if repeat else "")
            writer = DiagnosticWriter(root / label, plan, side=side,
                                      provenance=provenance[side], execution=execution)
            with ModuleCapture(model, writer, {str(i): [n] for i, n in enumerate(ids)},
                               inputs=lambda m, a, kw: {"hidden": a[0]},
                               outputs=lambda m, a, kw, y: {"hidden": y},
                               state=lambda m: {}, state_complete=True):
                model(inputs.clone())
            paths[label] = writer.finish()
    publish_json(root / "matrix-input.json", {"schema": "diagnostic-matrix.v1", "plan": "plan.json",
                 "cases": [{"id": "synthetic-injected-drift", **{k: str(v.relative_to(root)) for k, v in paths.items()}}]})
    result = run_matrix(root / "matrix-input.json", root / "report")
    endpoints = [ReplayEndpoint(call=lambda values, model=model: {"hidden": model[1](values["hidden"])},
                               reset=lambda: None, state=lambda: {}, provenance=provenance[side],
                               execution=execution, state_complete=True)
                 for side, model in (("left", left), ("right", right))]
    replay = replay_boundary(plan, paths["left"], paths["right"], ids[1], inputs={"hidden": inputs},
                             left_endpoint=endpoints[0], right_endpoint=endpoints[1], output=root / "replay")
    return {**result, "synthetic_only": True, "replay": str(root / "replay/replay.json"),
            "replay_verdict": replay["verdict"], "note": "Injected intermediate drift reconverges at the final output."}

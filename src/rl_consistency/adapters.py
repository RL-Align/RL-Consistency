"""Protocols for new model/engine adapters and a pinned Qwen3 launch bridge."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Protocol
from .common import digest, file_hash, read_json, require
from .contracts import validate_recipe

RLK_COMMIT = "76a127c206b7ead6fb837b46264ba43f9ef20e82"


class ModelAdapter(Protocol):
    def inventory(self) -> dict: ...
    def reduced_spec(self, selected_layers: list[int]) -> dict: ...


class EngineAdapter(Protocol):
    def provenance(self) -> dict: ...
    def capture(self, fixture: dict, weight_version: str) -> dict: ...


def plan(recipe, root, profile, python, *, mode="consistency", steps=2):
    validate_recipe(recipe)
    require(recipe["model"]["id"] == "Qwen/Qwen3-8B" and recipe["model"]["family"] == "qwen3",
            "launch bridge supports Qwen3-8B only; Qwen3.8-27B uses prepare-model and needs Hybrid runtime integration")
    require(recipe["rl_kernel"]["commit"] == RLK_COMMIT, "bridge has only been inspected against the pinned RL-Kernel commit")
    target = recipe["target"]
    require(target["training"]["engine"] == "megatron" and target["rollout"]["engine"] == "vllm"
            and recipe["orchestrator"] == "vime", "unsupported engine/orchestrator combination")
    require(mode in {"native", "consistency"}, "invalid execution mode")
    require(type(steps) is int and steps >= 2, "use at least two steps")
    require(target["world_size"] == 8, "initial Qwen3 bridge requires eight GPUs")
    train, rollout = target["training"], target["rollout"]
    require(train["tp"] in (1, 2, 4, 8) and train["tp"] * train["cp"] == 8, "unsupported training TP/CP")
    require(rollout["tp"] in (1, 2, 4, 8) and rollout["cp"] in (1, 2, 4, 8)
            and 8 % (rollout["tp"] * rollout["cp"]) == 0, "unsupported rollout TP/CP")
    for side in (train, rollout):
        require(side["pp"] == side["ep"] == side["dp"] == 1, "bridge does not support PP/EP/explicit DP")
    sample = recipe["sampling"]
    if target["backend"] == "rocm":
        require(rollout["cp"] == 1 and sample["temperature"] > 0 and sample["top_k"] == -1,
                "ROCm requires rollout CP1, positive temperature and top-k disabled")
    root, profile = Path(root).resolve(), Path(profile).resolve()
    # Use the canonical script with explicit Python and profile; avoid wrapper profile ambiguity.
    command = [str(Path(python).absolute()), str(root / "rl_engine/entrypoints/repro.py"),
               "run", "--profile", str(profile), "--backend", target["backend"], "--mode", mode,
               "--tp", str(train["tp"]), "--cp", str(train["cp"]),
               "--rollout-tp", str(rollout["tp"]), "--rollout-cp", str(rollout["cp"]),
               "--temperature", str(sample["temperature"]), "--top-p", str(sample["top_p"]),
               "--top-k", str(sample["top_k"]), "--steps", str(steps), "--wait",
               "--seed", str(recipe["inputs"]["seed"]), "--rollout-seed", str(recipe["inputs"]["seed"])]
    for key in ("lr", "weight_decay", "kl_coef", "max_response_len", "max_tokens_per_gpu"):
        command.extend(["--" + key.replace("_", "-"), str(recipe["benchmark"][key])])
    return {"schema": "rl-consistency.plan.v1", "recipe_sha256": digest(recipe),
            "cwd": str(root), "argv": command,
            "mode": mode, "certified": False,
            "note": "Delegates to RL-Kernel; successful exit does not pass C0-C10 or certify this recipe."}


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, timeout=15).stdout.strip()


def preflight(recipe, root, profile, python):
    validate_recipe(recipe, bound=True)
    root, profile = Path(root).resolve(), Path(profile).resolve()
    require(_git(root, "rev-parse", "HEAD") == recipe["rl_kernel"]["commit"], "RL-Kernel checkout revision differs")
    require(not _git(root, "status", "--porcelain"), "RL-Kernel checkout is dirty")
    require(file_hash(profile) == recipe["environment"]["profile_sha256"], "profile hash differs")
    configuration = read_json(profile)
    require(configuration.get("schema_version") == "rlkernel.repro.profile.v1", "invalid RL-Kernel profile")
    require(configuration.get("requirements", {}).get("backend", "cuda") == recipe["target"]["backend"], "profile backend differs")
    paths = configuration.get("paths", {})
    require(Path(paths.get("rl_kernel_root", str(root))).resolve() == root, "profile selects another RL-Kernel checkout")
    configured_python = paths.get("python")
    require(not configured_python or Path(configured_python).absolute() == Path(python).absolute(),
            "profile Python differs from explicit Python (venv identity is significant)")
    require(paths.get("model_root"), "explicit profile model_root required")
    model_config = Path(paths["model_root"]).expanduser() / "config.json"
    require(file_hash(model_config) == recipe["model"]["config_sha256"], "model config differs")
    for name, path_key in (("vime", "vime_root"), ("vllm", "vllm_root"), ("megatron", "megatron_root")):
        require(paths.get(path_key), f"explicit {path_key} required for preflight")
        source = Path(paths[path_key]).expanduser().resolve()
        require(_git(source, "rev-parse", "HEAD") == recipe["environment"]["sources"][name], f"{name} checkout differs")
        require(not _git(source, "status", "--porcelain"), f"{name} checkout is dirty")
    code = '''import importlib.metadata as m, json, platform, torch, rl_engine
packages = {}
for name in ("torch", "vllm", "ray", "triton", "transformer-engine"):
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = "missing"
devices = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    arch = getattr(p, "gcnArchName", "").split(":")[0] if torch.version.hip else "sm%d%d" % torch.cuda.get_device_capability(i)
    devices.append({"name": p.name, "architecture": arch})
print(json.dumps({"lock": {"python_version": platform.python_version(), "packages": packages},
 "backend": "rocm" if torch.version.hip else "cuda", "devices": devices,
 "rl_engine_file": rl_engine.__file__}))'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), *[str(paths[k]) for k in ("vime_root", "vllm_root", "megatron_root")]])
    result = subprocess.run([str(python), "-c", code], cwd=root, env=env, check=True,
                            capture_output=True, text=True, timeout=60)
    observed = json.loads(result.stdout)
    require(Path(observed["rl_engine_file"]).resolve().is_relative_to(root), "Python imported another RL-Kernel")
    require(digest(observed["lock"]) == recipe["environment"]["lock_sha256"], "Python package lock differs")
    require(observed["lock"]["python_version"] == recipe["environment"]["python_version"], "Python version differs")
    require(observed["backend"] == recipe["target"]["backend"], "active torch backend differs")
    require(len(observed["devices"]) == recipe["target"]["world_size"], "GPU count differs")
    require(all(d["architecture"] == recipe["target"]["architecture"] for d in observed["devices"]), "GPU architecture differs")
    return {"passed": True, "observed": observed,
            "note": "Host preflight only; workers must still record actual weights, binaries and operator bindings."}


def launch(recipe, root, profile, python, *, mode="consistency", steps=2, output):
    execution = plan(recipe, root, profile, python, mode=mode, steps=steps)
    preflight_result = preflight(recipe, root, profile, python)
    from .common import write_json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", execution)
    write_json(output / "preflight.json", preflight_result)
    # Pass one explicit profile and deterministic source lookup to the launcher.
    env = os.environ.copy()
    env["RLK_REPRO_PROFILE"] = str(Path(profile).resolve())
    env["PYTHONPATH"] = str(Path(root).resolve())
    with (output / "launcher.log").open("w") as log:
        process = subprocess.run(execution["argv"], cwd=execution["cwd"], env=env,
                                 stdout=log, stderr=subprocess.STDOUT, check=False)
    value = {"exit_code": process.returncode, "certified": False,
             "note": "Import worker evidence through gate commands; launcher success alone grants no gate."}
    write_json(output / "launcher-result.json", value)
    return value

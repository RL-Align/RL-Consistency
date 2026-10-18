"""CPU-only model onboarding bundle. Never emits successful GPU evidence."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from .common import file_hash, read_json, require, write_json
from .models import get_adapter


def inspect(model_id, config_path):
    adapter = get_adapter(model_id, read_json(config_path))
    result = adapter.inventory()
    result["config_sha256"] = file_hash(config_path)
    result["integration"] = {"level": "onboarding", "strict_launcher": False,
                             "gpu_verified": False, "runnable_diagnostic": False}
    return result


def prepare(model_id, config_path, backend, output, *, proxy_layers=None):
    require(backend in {"cuda", "rocm"}, "select cuda or rocm")
    adapter = get_adapter(model_id, read_json(config_path))
    if proxy_layers is None:
        proxy_layers = adapter.default_proxy_layers
    adapter.validate_depth(proxy_layers)
    proxy = adapter.reduced_spec(list(range(proxy_layers)))
    inventory = inspect(model_id, config_path)
    recipe_name = f"{adapter.recipe_prefix}-{backend}"
    recipe = json.loads(files("rl_consistency").joinpath("recipes", recipe_name+".json").read_text())
    recipe["model"].update(config_sha256=inventory["config_sha256"],
                            config_canonical_sha256=inventory["config_canonical_sha256"],
                            proxy_source_layers=proxy["selected_layers"])
    from .contracts import validate_recipe
    validate_recipe(recipe)
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/"proxy").mkdir()
    write_json(root/"source-config.json", adapter.config)
    write_json(root/"recipe.json", recipe)
    write_json(root/"inventory.json", inventory)
    write_json(root/"proxy-plan.json", proxy)
    write_json(root/"proxy"/"config.json", proxy["config"])
    write_json(root/"vime-model-args.json", {"vime_commit": adapter.vime_commit,
               "full_model": adapter.vime_model_args(), "proxy": adapter.vime_model_args(proxy_layers),
               "note": "Model args only; no launch command or runtime provider is selected."})
    result = {"schema": "rl-consistency.model-preparation.v1", "output": str(root),
              "model_id": model_id, "backend": backend, "status": "prepared-not-executed",
              "certified": False, "runnable_diagnostic": False, "proxy_layers": proxy_layers,
              "gates": {f"C{i}": "not-run" for i in range(11)},
              "blockers": [*adapter.blockers,
                           "proxy weights from the user's checkpoint (configuration is prepared only)",
                           "real baseline, C0-C10 GPU evidence and reviewed environment"],
              "next": f"Read skills/train-inference-consistency/references/{adapter.guide}; do not use the Qwen3-8B launch bridge.",
              "files": {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.json"))}}
    write_json(root/"preparation.json", result)
    return result

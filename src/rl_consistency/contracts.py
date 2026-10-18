"""Recipe identity and declared verification coverage; no implicit certification."""
from __future__ import annotations

import math
import re
from .common import ContractError, is_hash, positive_int, require

SCHEMA = "rl-consistency.recipe.v2"
BITWISE_GATES = ("C3", "C5", "C8", "C9")


def validate_recipe(recipe, bound=False):
    require(isinstance(recipe, dict), "recipe must be an object")
    require(recipe.get("schema") == SCHEMA, "recipe v2 required; v1 gate meanings must not be relabeled")
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]*", recipe.get("id", "")) is not None,
            "recipe id must be lowercase words separated by hyphens")
    require(recipe.get("status") in {"candidate", "verified"}, "invalid recipe status")
    model, target = recipe["model"], recipe["target"]
    require(bool(model.get("id")) and bool(model.get("family")), "model identity missing")
    require(target["backend"] in {"cuda", "rocm"}, "initial adapter supports CUDA/ROCm only")
    require(bool(target.get("architecture")) and positive_int(target["world_size"]),
            "hardware architecture/world size missing")
    for side in ("training", "rollout"):
        require(bool(target[side].get("engine")), f"{side} engine missing")
        for axis in ("tp", "cp", "pp", "ep", "dp"):
            require(positive_int(target[side][axis]), f"invalid {side}.{axis}")
    comparison = recipe["comparison"]
    require(comparison.get("kind") == "within_backend_train_rollout", "unsupported comparison scope")
    require(comparison.get("independent_recompute") is True, "logprob reuse cannot prove parity")
    require(comparison.get("nonfinite") == "reject", "non-finite outputs must fail")
    require(comparison.get("dtype") in {"float16", "bfloat16", "float32", "float64"}, "invalid output dtype")
    require(bool(comparison.get("scoring_semantics")), "logprob scoring semantics missing")
    require(positive_int(model.get("full_layers")), "full model layer count missing")
    for key in ("min_compared_elements", "max_context_tokens"):
        require(positive_int(recipe["inputs"].get(key)), f"inputs.{key} must be positive")
    require(type(recipe["inputs"].get("seed")) is int, "seed must be an integer")
    sample = recipe["sampling"]
    for key in ("temperature", "top_p"):
        require(type(sample.get(key)) in (int, float) and math.isfinite(sample[key]), f"invalid sampling.{key}")
    require(sample["temperature"] >= 0 and 0 < sample["top_p"] <= 1, "invalid sampling range")
    require(type(sample.get("top_k")) is int and (sample["top_k"] == -1 or sample["top_k"] > 0), "invalid top_k")
    for key in ("lr", "weight_decay", "kl_coef"):
        value = recipe["benchmark"].get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"invalid benchmark.{key}")
    for key in ("max_response_len", "max_tokens_per_gpu"):
        require(positive_int(recipe["benchmark"].get(key)), f"invalid benchmark.{key}")
    require(recipe["benchmark"].get("scope") == "end_to_end_step_replayed_tokens", "declare controlled performance scope")
    coverage = recipe["coverage"]
    for gate in (*BITWISE_GATES, "C7"):
        cases = coverage.get(gate)
        require(isinstance(cases, list) and len(cases) > 0 and
                all(isinstance(c, str) and c.strip() for c in cases) and len(cases) == len(set(cases)),
                f"{gate} requires unique, nonempty case IDs")
    for gate, required in {
        "C3": {"repeat", "batch", "padding", "permutation", "chunk", "prefill-decode", "mixed-scheduling", "cache-state"},
        "C5": {"old-logprob", "actor-forward"},
        "C7": {"forward-accuracy", "gradient-accuracy", "optimizer-accuracy"},
        "C8": {"theta-0", "theta-1", "theta-2"},
        "C9": {"old-logprob", "actor-forward", "theta-0", "theta-1", "theta-2", "long-context", "checkpoint-resume"},
    }.items():
        require(required <= set(coverage[gate]), f"{gate} misses mandatory case IDs")
    require("checkpoint-resume" in coverage["C9"], "C9 requires checkpoint-resume")
    tolerance = recipe["accuracy"]
    for key in ("atol", "rtol"):
        value = tolerance[key]
        require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
                f"accuracy.{key} must be finite and nonnegative")
    require(isinstance(recipe["operators"], list) and recipe["operators"] and
            all(isinstance(x, str) and x for x in recipe["operators"]) and
            len(recipe["operators"]) == len(set(recipe["operators"])), "operator inventory invalid")
    require(is_hash(recipe["rl_kernel"]["commit"], 40), "RL-Kernel commit must be pinned")
    if bound:
        budget = recipe.get("acceptance", {}).get("training_step_ratio_max")
        require(type(budget) in (int, float) and math.isfinite(budget) and budget > 0,
                "bind reviewed acceptance.training_step_ratio_max")
        for key in ("revision", "weights_sha256", "master_weights_sha256", "tokenizer_sha256", "config_sha256"):
            require(bool(model.get(key)), f"bind model.{key} before initializing a run")
        for key in ("weights_sha256", "master_weights_sha256", "tokenizer_sha256", "config_sha256"):
            require(is_hash(model[key]), f"model.{key} must be SHA256")
        require(is_hash(recipe["inputs"]["fixture_sha256"]), "bind inputs.fixture_sha256")
        environment = recipe.get("environment", {})
        require(is_hash(environment.get("lock_sha256")), "bind environment.lock_sha256")
        require(is_hash(environment.get("profile_sha256")), "bind environment.profile_sha256")
        require(bool(environment.get("python_version")), "bind environment.python_version")
        for name in ("vime", "vllm", "megatron"):
            require(is_hash(environment.get("sources", {}).get(name), 40), f"pin {name} commit")
    return recipe


def unbound_fields(recipe):
    fields = []
    for section, names in {
        "model": ("revision", "weights_sha256", "master_weights_sha256", "tokenizer_sha256", "config_sha256"),
        "acceptance": ("training_step_ratio_max",),
        "inputs": ("fixture_sha256",),
        "environment": ("lock_sha256", "profile_sha256", "python_version"),
    }.items():
        fields.extend(f"{section}.{name}" for name in names if not recipe.get(section, {}).get(name))
    fields.extend(f"environment.sources.{name}" for name in ("vime", "vllm", "megatron")
                  if not recipe.get("environment", {}).get("sources", {}).get(name))
    return fields

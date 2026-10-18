"""Machine checks for the workflow. Imported evidence is not a trusted attestation.

Producers must capture artifacts inside real workers. These validators catch
incomplete/inconsistent evidence; human review establishes producer trust.
"""
from __future__ import annotations

import math
from .common import require, positive_int, is_hash, digest
from .compare import compare_traces
from .contracts import validate_recipe
from .models.qwen38 import is_hybrid_recipe, check_model_evidence

TITLES = ["Module inventory and route", "Recipe and strong baseline", "Representative proxy",
          "Inference BI/PD", "Structural performance", "Dual-path training forward parity",
          "Backward and gradient contract", "Independent gradient validation",
          "Proxy two-update loop", "Reviewed full-model acceptance", "Optimization and delivery"]
DEPENDENCIES = {"C0": (), "C1": ("C0",), "C2": ("C0",), "C3": ("C2",),
                "C4": ("C3",), "C5": ("C4",), "C6": ("C5",), "C7": ("C6",),
                "C8": ("C7",), "C9": ("C8",), "C10": ("C1", "C9")}


def check_runtime(value, recipe):
    require(value.get("clean") is True, "formal evidence requires clean sources")
    require(value.get("rl_kernel_commit") == recipe["rl_kernel"]["commit"], "RL-Kernel revision mismatch")
    require(value.get("sources") == recipe["environment"]["sources"], "framework revision mismatch")
    for key in ("lock_sha256", "profile_sha256", "python_version"):
        require(value.get(key) == recipe["environment"][key], f"runtime {key} mismatch")
    for key in ("backend", "architecture", "world_size", "training", "rollout"):
        require(value.get(key) == recipe["target"][key], f"runtime {key} mismatch")
    loaded = value.get("loaded_artifacts")
    require(isinstance(loaded, list) and loaded, "worker-loaded source/binary fingerprints missing")
    for item in loaded:
        require(isinstance(item.get("path"), str) and item["path"] and is_hash(item.get("sha256")),
                "loaded artifact must have path and hash")
    roles = {item.get("worker") for item in loaded}
    require({"rollout", "training"} <= roles, "collect provenance from both worker roles")


def benchmark(value, recipe):
    require(value.get("fixture_sha256") == recipe["inputs"]["fixture_sha256"], "benchmark fixture mismatch")
    require(value.get("model_sha256") == recipe["model"]["weights_sha256"], "benchmark model mismatch")
    require(value.get("target") == recipe["target"], "benchmark target mismatch")
    require(value.get("settings") == recipe["benchmark"], "benchmark settings mismatch")
    require(value.get("sampling") == recipe["sampling"], "benchmark sampling semantics mismatch")
    times = value.get("step_seconds")
    require(isinstance(times, list) and len(times) >= 3, "benchmark needs at least 3 measured steps")
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in times), "invalid timing")
    require(positive_int(value.get("total_output_tokens")), "benchmark output-token count missing")
    require(positive_int(value.get("peak_memory_bytes")), "benchmark peak memory missing")
    require(type(value.get("warmup_steps")) is int and value["warmup_steps"] >= 1, "exclude warmup steps")
    require(is_hash(value.get("execution_sha256")), "actual execution fingerprint missing")
    return {"mean_step_seconds": sum(times) / len(times),
            "tokens_per_second": value["total_output_tokens"] / sum(times),
            "peak_memory_bytes": value["peak_memory_bytes"], "steps": len(times)}


def pairs(gate, evidence, recipe, artifact):
    items = evidence.get("pairs", [])
    ids = [x["case"] for x in items]
    require(len(ids) == len(set(ids)) and set(ids) == set(recipe["coverage"][gate]),
            f"{gate} must contain exactly the declared case matrix")
    results = []
    for pair in items:
        require(pair["left"] != pair["right"], "cannot compare an artifact with itself")
        left, right = artifact(pair["left"]), artifact(pair["right"])
        require(left["inputs_sha256"] == recipe["inputs"]["fixture_sha256"], "trace fixture mismatch")
        require(left["semantics_sha256"] == digest(recipe["comparison"]), "trace scoring contract mismatch")
        if gate in {"C3", "C5", "C7"}:
            require(left["weights_sha256"] == recipe["model"]["weights_sha256"], "initial weights mismatch")
        if gate != "C7":
            require(left["dtype"] == recipe["comparison"]["dtype"], "trace output dtype differs from recipe")
        if gate in {"C5", "C8", "C9"}:
            require(left["producer"] == "rollout" and right["producer"] == "training",
                    "expected independently computed rollout/training traces")
        if gate in {"C5", "C9"} and pair["case"] in {"old-logprob", "actor-forward"}:
            receipt = right.get("execution_path", {})
            require(receipt.get("kind") == pair["case"] and receipt.get("forward_executed") is True
                    and positive_int(receipt.get("calls")) and is_hash(receipt.get("callsite_sha256")),
                    "native training callsite evidence missing")
            require(receipt.get("model_mode") in {"train", "eval"}
                    and type(receipt.get("grad_enabled")) is bool
                    and type(receipt.get("output_requires_grad")) is bool, "training execution modes missing")
            if pair["case"] == "actor-forward":
                require(receipt["model_mode"] == "train" and receipt["grad_enabled"]
                        and receipt["output_requires_grad"], "actor forward must execute with autograd in train mode")
            require(left["weights_sha256"] == recipe["model"]["weights_sha256"], "theta-0 entry weights mismatch")
        if gate == "C7":
            require(left["producer"] == "training" and right["producer"] == "reference", "gradient reference missing")
            require(left["implementation_sha256"] != right["implementation_sha256"], "gradient reference reuses implementation")
            require(right.get("reference_kind") == "independent_mathematical", "independent mathematical reference required")
        result = compare_traces(left, right, accuracy=gate == "C7", **recipe["accuracy"])
        require(result["elements"] >= recipe["inputs"]["min_compared_elements"], "insufficient comparison coverage")
        results.append({"case": pair["case"], **result, "weight_version": left["weight_version"],
                        "weights_sha256": left["weights_sha256"]})
    return results


def check_loop(evidence, recipe, artifact, comparisons):
    """Two effective master updates, with same-policy comparisons at theta0/1/2."""
    updates, loop = artifact(evidence["updates"]), artifact(evidence["loop"])
    require(len(updates) == 2, "short acceptance requires exactly two effective updates")
    require(loop.get("normal_exit") is True and loop.get("executed_updates") == 2
            and isinstance(loop.get("framework_entrypoint"), str) and loop["framework_entrypoint"],
            "real framework loop and normal exit required")
    by_case = {x["case"]: x for x in comparisons}
    initial = by_case["theta-0"]
    require(initial["weights_sha256"] == recipe["model"]["weights_sha256"], "theta-0 comparison missing")
    previous, master = initial["weights_sha256"], recipe["model"]["master_weights_sha256"]
    versions = {initial["weight_version"]}
    for step, update in enumerate(updates, 1):
        require(update["before_sha256"] == previous and is_hash(update["after_sha256"]), "broken deployment weight chain")
        require(update.get("before_master_sha256") == master and is_hash(update.get("after_master_sha256"))
                and update["after_master_sha256"] != master, "master parameters did not change")
        for key in ("master_update_norm", "gradient_norm"):
            value = update.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, f"invalid {key}")
        require(update.get("finite_loss") is True and update.get("gradient_coverage_complete") is True
                and update.get("optimizer_step_executed") is True and type(update.get("optimizer_step")) is int
                and update["optimizer_step"] == step, "effective optimizer evidence missing")
        require(is_hash(update.get("optimizer_state_sha256")), "optimizer state evidence missing")
        match = by_case[f"theta-{step}"]
        require(update["version"] not in versions and match["weight_version"] == update["version"]
                and match["weights_sha256"] == update["after_sha256"], "post-sync theta comparison missing")
        versions.add(update["version"])
        previous, master = update["after_sha256"], update["after_master_sha256"]
    return {"updates": updates, "loop": loop}


def evaluate(gate, evidence, recipe, artifact, baseline=None):
    validate_recipe(recipe, bound=True)
    require(gate in DEPENDENCIES, "unknown gate")
    require(evidence.get("schema") == "rl-consistency.evidence.v2", "evidence v2 required; old gate meanings cannot be reused")
    require(evidence.get("gate") == gate, "evidence gate mismatch")
    require(evidence.get("recipe_sha256") == digest(recipe), "evidence belongs to a different recipe")
    require(evidence.get("environment_sha256") == digest(recipe["environment"]), "evidence environment mismatch")
    require(evidence.get("execution") == "real", "synthetic/smoke data cannot pass model acceptance")
    if is_hybrid_recipe(recipe):
        check_model_evidence(gate, evidence, recipe, artifact)
    detail = {}
    if gate == "C0":
        runtime = artifact(evidence["runtime"])
        check_runtime(runtime, recipe)
        require(runtime.get("model") == recipe["model"], "worker model identity mismatch")
        require(runtime.get("fixture_sha256") == recipe["inputs"]["fixture_sha256"], "worker fixture mismatch")
        inventory = artifact(evidence["inventory"])
        modules = inventory.get("modules", [])
        require(inventory.get("route") == recipe["target"]["training"]["engine"], "training route mismatch")
        require(len(modules) == len(recipe["operators"]) and {m["operator"] for m in modules} == set(recipe["operators"]),
                "module inventory incomplete")
        for module in modules:
            require(all(isinstance(module.get(k), str) and module[k] for k in
                        ("inference_implementation", "training_math", "quantization", "discrete_choices")),
                    "module arithmetic/quantization/selection inventory missing")
        detail = {"runtime": runtime, "inventory": inventory}
    elif gate == "C1":
        value = artifact(evidence["benchmark"])
        require(value.get("mode") == "native", "C1 requires a native baseline")
        detail = {"benchmark": value, "summary": benchmark(value, recipe)}
    elif gate == "C2":
        value = artifact(evidence["coverage"])
        require(value.get("source_config_sha256") == recipe["model"]["config_sha256"], "reduced model source differs")
        source = value["source_layers"]
        selected = value["selected_layers"]
        require(len(source) == recipe["model"]["full_layers"], "source layer count differs")
        require(len(selected) >= 2 and len(selected) <= len(source), "invalid reduced layer count")
        require(all(type(i) is int and 0 <= i < len(source) for i in selected) and
                selected == sorted(set(selected)), "selected layers must be unique and ordered")
        reduced = value["reduced_layers"]
        require(reduced == [source[i] for i in selected], "reduced model changed dimensions, quantization or layer semantics")
        for layer in source:
            require(layer.get("operators") and layer.get("dimensions") and layer.get("precision"), "incomplete layer specification")
        all_ops = {op for layer in source for op in layer["operators"]}
        covered = {op for layer in reduced for op in layer["operators"]}
        require(all_ops <= covered, "reduced model misses operator types")
        kinds = [digest(layer) for layer in source]
        kept = [kinds[i] for i in selected]
        require(set(zip(kinds, kinds[1:])) <= set(zip(kept, kept[1:])), "reduced model misses layer interactions")
        require(value.get("boundary_operators") == recipe["operators"], "embedding/head and operator boundary coverage missing")
        detail = {"selected_layers": selected, "operators": sorted(covered)}
    elif gate == "C4":
        profile = artifact(evidence["profile"])
        require(is_hash(profile.get("raw_profile_sha256")) and profile.get("target") == recipe["target"],
                "profile identity/target missing")
        require(profile.get("per_token_python_loop") is False and profile.get("unbounded_workspace") is False
                and profile.get("graph_capture_replay") is True, "structural execution defect or missing Graph evidence")
        shapes = profile.get("operator_shapes", [])
        require({x["operator"] for x in shapes} == set(recipe["operators"]), "profile operator coverage incomplete")
        for item in shapes:
            require(item.get("shape") and all(positive_int(x) for x in item["shape"])
                    and positive_int(item.get("calls")), "real shape/call-count profile missing")
        require(positive_int(profile.get("peak_memory_bytes")), "profile peak memory missing")
        detail = {"profile": profile}
    elif gate == "C6":
        backward = artifact(evidence["backward"])
        modules = backward.get("modules", [])
        require(len(modules) == len(recipe["operators"]) and {x["operator"] for x in modules} == set(recipe["operators"]),
                "backward inventory incomplete")
        for item in modules:
            require(item.get("strategy") in {"reuse", "glue", "custom"} and item.get("rationale")
                    and item.get("saved_from_forward") and is_hash(item.get("gradient_contract_sha256")),
                    "backward/STE contract or saved-state mapping missing")
        timing = backward.get("timing", {})
        require(timing.get("target") == recipe["target"], "training timing topology mismatch")
        for key in ("native_step_seconds", "strict_step_seconds"):
            value = timing.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0, "training timing missing")
        ratio = timing["strict_step_seconds"] / timing["native_step_seconds"]
        require(ratio <= recipe["acceptance"]["training_step_ratio_max"], "training step exceeds reviewed budget")
        detail = {"backward": backward, "training_step_ratio": ratio}
    elif gate == "C5":
        runtime = artifact(evidence["runtime"])
        check_runtime(runtime, recipe)
        bindings = artifact(evidence["bindings"])
        expected = {(side, op) for side in ("training", "rollout") for op in recipe["operators"]}
        actual = [(x["side"], x["operator"]) for x in bindings]
        require(len(actual) == len(set(actual)) and set(actual) == expected, "runtime operator coverage incomplete")
        for value in bindings:
            require(value.get("fallback") is False and value.get("observable") is True, "fallback or unobservable route")
            require(is_hash(value.get("implementation_sha256")) and is_hash(value.get("contract_sha256")), "binding fingerprint missing")
        for op in recipe["operators"]:
            contracts = {b["contract_sha256"] for b in bindings if b["operator"] == op}
            require(len(contracts) == 1, f"numerical contracts differ: {op}")
        detail = {"bindings": bindings, "runtime": runtime,
                  "comparisons": pairs(gate, evidence, recipe, artifact)}
    elif gate in {"C3", "C7", "C8", "C9"}:
        detail = {"comparisons": pairs(gate, evidence, recipe, artifact)}
        if gate in {"C8", "C9"}:
            detail.update(check_loop(evidence, recipe, artifact, detail["comparisons"]))
        if gate == "C9":
            full = artifact(evidence["full_model"])
            require(full.get("config_sha256") == recipe["model"]["config_sha256"] and
                    full.get("layers") == recipe["model"]["full_layers"], "C9 requires the unmodified full model")
            require(type(full.get("context_tokens")) is int and full["context_tokens"] >= recipe["inputs"]["max_context_tokens"],
                    "full-model context coverage insufficient")
            require(full.get("target") == recipe["target"], "full-model topology differs")
            checkpoint = artifact(evidence["checkpoint"])
            for key in ("model", "optimizer", "scheduler", "rng", "data_cursor"):
                a = checkpoint["uninterrupted_state"].get(key)
                b = checkpoint["restored_state"].get(key)
                require(is_hash(a) and a == b, f"restored {key} state differs")
            require(checkpoint.get("resumed_first_step") is True, "checkpoint save alone does not pass")
            # Resume equivalence is a separate comparison from train/rollout parity.
            control_trace = artifact(checkpoint["uninterrupted_trace"])
            resumed_trace = artifact(checkpoint["resumed_trace"])
            require(checkpoint["uninterrupted_trace"] != checkpoint["resumed_trace"], "resume requires separate captures")
            require(resumed_trace["weights_sha256"] == checkpoint["restored_state"]["model"], "resumed trace uses different weights")
            require(resumed_trace["inputs_sha256"] == recipe["inputs"]["fixture_sha256"] and
                    resumed_trace["semantics_sha256"] == digest(recipe["comparison"]), "resumed trace identity differs")
            resumed_pair = next(x for x in detail["comparisons"] if x["case"] == "checkpoint-resume")
            require(resumed_pair["weights_sha256"] == resumed_trace["weights_sha256"] and
                    resumed_pair["weight_version"] == resumed_trace["weight_version"], "resume parity pair uses another version")
            resume = compare_traces(control_trace, resumed_trace)
            require(resume["passed"], "uninterrupted/resumed first-step outputs differ")
            require(is_hash(checkpoint.get("uninterrupted_after_update_sha256")) and
                    checkpoint["uninterrupted_after_update_sha256"] == checkpoint.get("resumed_after_update_sha256"),
                    "first resumed update differs from uninterrupted execution")
            detail["resume"] = resume
    elif gate == "C10":
        require(baseline is not None, "frozen baseline missing")
        value = artifact(evidence["benchmark"])
        require(value.get("mode") == "consistency", "C10 requires strict benchmark")
        a, b = benchmark(baseline, recipe), benchmark(value, recipe)
        require(a["steps"] == b["steps"] and baseline["total_output_tokens"] == value["total_output_tokens"],
                "controlled benchmark requires equal steps and replayed output token work")
        detail = {"baseline": a, "consistency": b,
                  "step_overhead_percent": (b["mean_step_seconds"] / a["mean_step_seconds"] - 1) * 100,
                  "memory_ratio": b["peak_memory_bytes"] / a["peak_memory_bytes"],
                  "performance_threshold_applied": False}
    passed = all(x["passed"] for x in detail.get("comparisons", []))
    return {"gate": gate, "passed": passed, "detail": detail}

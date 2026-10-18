"""Runnable fixed-policy ROCm diagnostic using the inspected RL-Kernel entrypoint.

This runner does not synthesize C0-C10 receipts or certify a model. It keeps the
machine profile's workload, runs one round at LR=WD=0, and collects raw sidecars.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from .adapters import RLK_COMMIT, _git
from .common import contained, digest, file_hash, is_hash, positive_int, publish_json, read_json, require, write_json
from .gates import TITLES
from .rlk_import import inspect_sidecars

DEFAULTS = {
    "tp": 4, "cp": 2, "rollout-tp": 4, "rollout-cp": 1,
    "temperature": 1.0, "top-p": 1.0, "top-k": -1,
    "samples-per-prompt": 8, "global-batch-size": 8, "rollout-batch-size": 1,
    "max-response-len": 7168, "max-tokens-per-gpu": 4096,
    "vllm-gpu-memory-utilization": 0.4, "seed": 1234, "rollout-seed": 1234,
    "kl-coef": 0.0, "grpo-std-normalization": "enabled",
    "ray-port": 8090, "ray-dashboard-port": 48025,
}
PATHS = ("workspace", "rl_kernel_root", "vime_root", "megatron_root", "vllm_root",
         "python", "model_root", "ref_load", "prompt_data")


def plan(profile, output, *, mode="consistency"):
    profile = Path(profile).expanduser().resolve()
    config = read_json(profile)
    require(config.get("schema_version") == "rlkernel.repro.profile.v1", "expected RL-Kernel machine profile")
    require(config.get("requirements", {}).get("backend") == "rocm", "profile must select ROCm")
    require(mode in {"native", "consistency"}, "invalid ROCm mode")
    raw_paths = config.get("paths", {})
    paths = {}
    for name in PATHS:
        value = raw_paths.get(name)
        require(isinstance(value, str) and value, f"profile paths.{name} is required")
        path = Path(value).expanduser()
        require(path.is_absolute(), f"profile paths.{name} must be absolute")
        paths[name] = str(path.absolute() if name == "python" else path.resolve())
    supplied = copy.deepcopy(config.get("defaults", {}))
    require(isinstance(supplied, dict), "profile defaults must be an object")
    if "max-response-length" in supplied:
        require("max-response-len" not in supplied, "conflicting response length aliases")
        supplied["max-response-len"] = supplied.pop("max-response-length")
    require(not set(supplied) - (set(DEFAULTS) | {"backend", "mode", "steps", "rollouts", "lr", "weight-decay"}),
            "unknown profile defaults; review before running")
    if "backend" in supplied:
        require(supplied["backend"] == "rocm", "conflicting backend default")
    values = {**DEFAULTS, **{k:v for k,v in supplied.items() if k in DEFAULTS}}
    require(values["tp"] in (1,2,4,8) and positive_int(values["cp"]) and values["tp"]*values["cp"] == 8,
            "training TP*CP must equal eight")
    require(values["rollout-tp"] in (1,2,4,8) and values["rollout-cp"] == 1, "rollout TP1/2/4/8 and CP1 supported")
    for key in ("samples-per-prompt", "global-batch-size", "rollout-batch-size", "max-response-len", "max-tokens-per-gpu"):
        require(positive_int(values[key]), f"invalid {key}")
    require(type(values["tp"]) is int and type(values["rollout-tp"]) is int
            and type(values["rollout-cp"]) is int and type(values["top-k"]) is int, "integer topology and top-k required")
    require(values["global-batch-size"] == values["rollout-batch-size"]*values["samples-per-prompt"],
            "one-round diagnostic requires global batch = rollout batch * samples per prompt")
    require(values["global-batch-size"] >= 8//values["rollout-tp"], "each rollout engine needs a request")
    for key in ("temperature", "top-p", "vllm-gpu-memory-utilization", "kl-coef"):
        require(type(values[key]) in (int,float) and math.isfinite(values[key]), f"invalid {key}")
    require(values["temperature"] > 0 and 0 < values["top-p"] <= 1 and values["top-k"] == -1,
            "ROCm requires positive temperature, valid top-p and top-k=-1")
    require(0 < values["vllm-gpu-memory-utilization"] < 1 and values["kl-coef"] >= 0, "invalid memory/KL setting")
    require(values["grpo-std-normalization"] in {"enabled", "disabled"}, "invalid GRPO normalization")
    require(all(type(values[k]) is int for k in ("seed", "rollout-seed")), "integer seeds required")
    require(all(type(values[k]) is int and 1024 <= values[k] <= 65535 for k in ("ray-port", "ray-dashboard-port"))
            and values["ray-port"] != values["ray-dashboard-port"], "distinct unprivileged Ray ports required")
    output = Path(output).expanduser().resolve()
    # Only these diagnostic changes override the existing workload. No shortened
    # tokens/batch, precision change, hidden optimizer run or schedule compression.
    overrides = {"backend":"rocm", "mode":mode, "steps":1, "lr":0.0, "weight-decay":0.0}
    resolved = {"schema_version":config["schema_version"], "name":"rl-consistency-fixed-policy",
                "requirements":config["requirements"], "paths":{**paths,"output_root":str(output)},
                "defaults":{**values, **overrides}}
    command = [paths["python"], str(Path(paths["rl_kernel_root"])/"rl_engine/entrypoints/repro.py"),
               "run", "--profile", str(output/"profile.json"), "--run-id", "upstream", "--wait"]
    return {"schema":"rl-consistency.rocm-plan.v1", "source_profile":str(profile),
            "source_profile_sha256":file_hash(profile), "resolved_profile":resolved,
            "rl_kernel_commit":RLK_COMMIT, "output":str(output), "argv":command,
            "diagnostic_overrides":overrides, "expected_samples":values["global-batch-size"],
            "world_size":8, "architecture":"gfx942", "certified":False,
            "scope":"one-round fixed-policy selected-logprob diagnostic; not effective-update acceptance"}


def _environment(execution):
    paths = execution["resolved_profile"]["paths"]
    env = os.environ.copy()
    # Repro supports environment overrides; the saved profile must be authoritative.
    for name in list(env):
        if name.startswith("RLK_REPRO_"):
            del env[name]
    env["PYTHONPATH"] = os.pathsep.join(paths[k] for k in ("rl_kernel_root","vime_root","megatron_root","vllm_root"))
    env["PATH"] = str(Path(paths["python"]).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def preflight(execution):
    paths = execution["resolved_profile"]["paths"]
    for name in PATHS:
        path = Path(paths[name])
        require(path.exists(), f"missing {name}: {path}")
    require(_git(paths["rl_kernel_root"], "rev-parse", "HEAD") == RLK_COMMIT, "use the pinned RL-Kernel checkout")
    require(not _git(paths["rl_kernel_root"], "status", "--porcelain"), "pinned RL-Kernel checkout must be clean")
    sources = {}
    for key in ("rl_kernel_root", "vime_root", "megatron_root", "vllm_root"):
        root = paths[key]
        diff = _git(root, "diff", "HEAD", "--binary")
        sources[key] = {"head":_git(root,"rev-parse","HEAD"), "status":_git(root,"status","--porcelain"),
                        "tracked_diff_sha256":hashlib.sha256(diff.encode()).hexdigest()}
    model = read_json(Path(paths["model_root"])/"config.json")
    dimensions = {"num_hidden_layers":36, "hidden_size":4096, "intermediate_size":12288,
                  "num_attention_heads":32, "num_key_value_heads":8, "head_dim":128, "vocab_size":151936}
    require(model.get("model_type") == "qwen3" and all(model.get(k) == v for k,v in dimensions.items())
            and not model.get("quantization_config"),
            "this launch bridge requires the unquantized Qwen3-8B model config")
    code = '''import json, sys, torch, rl_engine, vllm, vime, megatron.core
items = []
for i in range(torch.cuda.device_count()):
 p = torch.cuda.get_device_properties(i)
 items.append({"name":p.name,"arch":getattr(p,"gcnArchName","").split(":")[0]})
print("RL_CONSISTENCY_PROBE="+json.dumps({"hip":torch.version.hip,"torch":torch.__version__,"prefix":sys.prefix,
 "python":sys.version,"devices":items,"modules":{m.__name__:m.__file__ for m in (rl_engine,vllm,vime,megatron.core)}}))'''
    process = subprocess.run([paths["python"],"-c",code], env=_environment(execution),
                             capture_output=True,text=True,check=True,timeout=120)
    lines = [line.split("=",1)[1] for line in process.stdout.splitlines() if line.startswith("RL_CONSISTENCY_PROBE=")]
    require(len(lines) == 1, "runtime probe did not return one result")
    observed = json.loads(lines[0])
    require(Path(observed["prefix"]).resolve() == Path(sys.prefix).resolve(),
            "run rl_consistency with the profile's Python environment so .pt collection uses the same PyTorch")
    require(observed["hip"] and len(observed["devices"]) == 8
            and all(d["arch"] == "gfx942" for d in observed["devices"]), "requires eight visible MI300X/gfx942 ROCm GPUs")
    for module, key in (("rl_engine","rl_kernel_root"),("vllm","vllm_root"),("vime","vime_root"),("megatron.core","megatron_root")):
        require(observed["modules"].get(module) and Path(observed["modules"][module]).resolve().is_relative_to(Path(paths[key])),
                f"runtime imported a different {module} checkout")
    return {"passed":True,"sources":sources,"runtime":observed,
            "model_config_sha256":file_hash(Path(paths["model_root"])/"config.json"),
            "prompt_data_sha256":file_hash(paths["prompt_data"]),
            "note":"Host observation only; companion patches are retained and reported, not installed or modified."}


def _html(root, result):
    rows = "".join(f"<tr><td>C{i}</td><td>{html.escape(title)}</td><td>not run</td></tr>" for i,title in enumerate(TITLES))
    payload = html.escape(json.dumps(result,indent=2,ensure_ascii=False))
    (root/"report.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>ROCm consistency diagnostic</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}
td,th{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>ROCm diagnostic: {html.escape(result['verdict'])}</h1>
<p>Original-dtype selected-logprob comparison for one fixed-policy run. No C0-C10 certification.</p>
<p><a href="diagnostic.json">Raw comparison</a> · <a href="launcher.log">Launcher log</a> · <a href="result.json">Result JSON</a></p>
<pre>{payload}</pre><table><tr><th>Gate</th><th>Meaning</th><th>Acceptance</th></tr>{rows}</table></html>''',encoding="utf-8")


def collect(root):
    root = Path(root).resolve()
    execution = read_json(root/"plan.json")
    values = execution["resolved_profile"]["defaults"]
    mode = values["mode"]; case = "R/R" if mode == "consistency" else "P/P"
    arm = "upstream/arms/"+case.lower().replace("/","-")
    result = {"schema":"rl-consistency.rocm-result.v1", "verdict":"inconclusive", "certified":False,
              "plan_sha256":digest(execution), "scope":execution["scope"], "mode":mode,
              "not_verified":["native old-logprob path", "grad-enabled actor execution receipt", "complete loaded weight identity",
                              "inference BI/PD matrix", "independent gradients", "theta0/theta1/theta2 effective-update loop",
                              "checkpoint resume", "quality", "performance", "C0-C10 acceptance"], "errors":[]}
    try:
        launcher = read_json(root/"launcher-result.json")
        summary = read_json(contained(root,"upstream/single-arm-summary.json"))
        validation = read_json(contained(root,arm+"/validation.json"))
        launch = read_json(contained(root,arm+"/launch.json"))
        result["upstream_errors"] = list(dict.fromkeys(summary.get("errors", []) + validation.get("errors", [])))
        diagnostic = inspect_sidecars(contained(root,arm+"/mismatch_sidecars"),
                                      tp=values["tp"],cp=values["cp"],world_size=8,layout="vime-zigzag")
        if (root/"diagnostic.json").exists():
            require(digest(read_json(root/"diagnostic.json")) == digest(diagnostic),
                    "sidecar/diagnostic changed since collection; start a new output directory")
        else:
            publish_json(root/"diagnostic.json",diagnostic)
        result["comparison"] = diagnostic["comparison"]
        result["sample_count"] = diagnostic["sample_count"]
        require(diagnostic["sample_count"] == execution["expected_samples"], "sample coverage differs from resolved workload")
        require(summary.get("case_id") == case and validation.get("case_id") == case and launch.get("case_id") == case,
                "executed implementation case differs from plan")
        require(summary.get("num_rollout") == 1 and validation.get("expected_sample_count") == execution["expected_samples"],
                "upstream rollout/sample counts differ")
        require(launch.get("framework_consistency", {}).get("use_rollout_logprobs") is False
                and launch["framework_consistency"].get("mismatch_metrics_recompute") is True, "rollout-logprob reuse enabled")
        env = launch.get("environment", {})
        require(float(env.get("RLK_ABLATION_LR", "nan")) == 0.0 and float(env.get("RLK_ABLATION_WEIGHT_DECAY", "nan")) == 0.0,
                "executed learning rate/decay differ from fixed-policy plan")
        require(summary.get("frozen_sources_match") is True and validation.get("frozen_sources_match") is True,
                "source freeze failed")
        before = read_json(contained(root,"upstream/frozen-inputs.before.json"))
        after = read_json(contained(root,"upstream/frozen-inputs.after.json"))
        require(is_hash(before.get("fingerprint")) and before["fingerprint"] == after.get("fingerprint")
                and before["fingerprint"] == launch.get("frozen_input_fingerprint"), "frozen manifest fingerprints differ")
        require(launcher.get("exit_code") == 0 and summary.get("launcher_returncode") == 0
                and validation.get("launcher_returncode") == 0, "launcher did not exit normally")
        require(summary.get("passed") is True and validation.get("passed") is True, "upstream validation failed")
        result["verdict"] = "pass" if diagnostic["comparison"]["passed"] else "fail"
        result["artifacts"] = {name:file_hash(contained(root,name)) for name in
            ("profile.json","preflight.json","diagnostic.json","launcher-result.json",arm+"/launch.json",arm+"/validation.json",
             "upstream/single-arm-summary.json","upstream/frozen-inputs.before.json","upstream/frozen-inputs.after.json")}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
        result["verdict"] = "fail" if result.get("comparison", {}).get("passed") is False else "inconclusive"
    write_json(root/"result.json",result)
    _html(root,result)
    return result


def run(profile, output, *, mode="consistency"):
    execution = plan(profile,output,mode=mode)
    root = Path(execution["output"])
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/"plan.json",execution)
    write_json(root/"profile.json",execution["resolved_profile"])
    try:
        write_json(root/"preflight.json",preflight(execution))
        print(f"ROCm fixed-policy run; engine log: {root}/upstream/arms/{'r-r' if mode == 'consistency' else 'p-p'}/launcher.log",
              file=sys.stderr,flush=True)
        with (root/"launcher.log").open("w") as log:
            process = subprocess.run(execution["argv"],env=_environment(execution),
                                     cwd=execution["resolved_profile"]["paths"]["rl_kernel_root"],
                                     stdout=log,stderr=subprocess.STDOUT,check=False)
        write_json(root/"launcher-result.json",{"exit_code":process.returncode})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        write_json(root/"launcher-result.json",{"exit_code":None,"error":str(exc)})
        write_json(root/"preflight-error.json",{"error":str(exc)})
    result = collect(root)
    if (root/"preflight-error.json").exists():
        result["errors"].insert(0,read_json(root/"preflight-error.json")["error"])
        write_json(root/"result.json",result); _html(root,result)
    return result

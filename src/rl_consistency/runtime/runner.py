"""Run scoped parity checks and optional real SGD calibration updates."""
from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
import platform

from ..common import ContractError, digest, file_hash, publish_json, read_json, require
from .models import load_checkpoint, validate_profile

SCOPE = {
    "runtime": "shared-prefix-v1", "physical_batch_size": 1, "kv_cache": False,
    "attention": "eager", "experts": "eager", "hub_kernels": False,
    "grad_enabled_in_all_phases": True, "model_mode": "eval",
    "logprob": "FP32 logit[token] - logsumexp(full vocabulary)",
    "generation": "greedy, fixed number of tokens (including after EOS)",
    "compared": ["decoder boundary hashes", "router logits/weights/ids", "vocabulary logits", "selected logprob"],
    "excluded": ["native batching", "KV-cache/prefill-decode parity", "multiple devices",
                 "vLLM/Megatron parity", "cross-hardware parity", "optimizer trajectory parity",
                 "independent mathematical accuracy", "checkpoint resume", "C0-C10 acceptance"],
}


def run_loaded(model, requests, output, *, identity, max_new_tokens=2, updates=0, learning_rate=0.01):
    """API for local model integrations. Results describe only the executed fixture."""
    import torch
    from .execution import Divergence, SharedPrefixRuntime, deterministic_execution, state_hash

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "rl-consistency.runtime-check.v1", "scope": SCOPE,
              "certified": False, "full_model_gpu_verified": False,
              "identity": identity, "fixture_sha256": digest(requests), "requests": requests,
              "python": platform.python_version(), "torch_threads": torch.get_num_threads(),
              "rounds": [], "updates": [], "verdict": "error"}
    result["implementation"] = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    try:
        require(type(updates) is int and 0 <= updates <= 2, "updates must be 0, 1 or 2 calibration steps")
        require(type(learning_rate) in {int, float} and math.isfinite(learning_rate) and learning_rate > 0,
                "finite positive learning rate required")
        runtime = SharedPrefixRuntime(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) if updates else None
        for theta in range(updates + 1):
            rollout = runtime.generate(requests, max_new_tokens=max_new_tokens)
            # Persist rollout before recomputing; later phases cannot replace it with their own values.
            rollout_path = output / f"theta-{theta}-rollout.json"
            publish_json(rollout_path, [asdict(t) for t in rollout])
            checked = runtime.check(rollout)
            check_path = output / f"theta-{theta}-check.json"
            publish_json(check_path, checked)
            result["rounds"].append({"theta": theta, "state_sha256": checked["state_sha256"],
                                     "rollout": {"path": rollout_path.name, "sha256": file_hash(rollout_path)},
                                     "check": {"path": check_path.name, "sha256": file_hash(check_path)},
                                     "generated": [{"id": t.request, "token_ids": t.tokens} for t in rollout],
                                     "parameters_with_gradient": checked["parameters_with_gradient"],
                                     "parameters_with_nonzero_gradient": checked["parameters_with_nonzero_gradient"]})
            if theta < updates:
                before = checked["state_sha256"]
                with deterministic_execution():
                    optimizer.step()
                after = state_hash(model)
                require(after != before, "SGD step did not change model bytes; calibration update is ineffective")
                result["updates"].append({"optimizer": "SGD", "learning_rate": learning_rate,
                                           "before_sha256": before, "after_sha256": after})
        result["verdict"] = "equal_observed"
    except Divergence as exc:
        result["verdict"] = "mismatch"
        result["divergence"] = exc.detail
    except (ContractError, RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        model.zero_grad(set_to_none=True)
        publish_json(output / "result.json", result)
    return result


def run(args):
    # Must be set before CUDA initialization. Existing incompatible values fail in the executor.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    require(not args.output.exists(), "output directory must be new")
    require(args.checkpoint.is_dir(), "checkpoint must be a local directory containing actual weights")
    family = validate_profile(args.model, read_json(args.checkpoint / "config.json"))
    model, identity = load_checkpoint(args.checkpoint, family, device=args.device, dtype=args.dtype)
    identity["model_profile"] = args.model
    if args.fixture:
        fixture = read_json(args.fixture)
        require(fixture.get("schema") == "rl-consistency.token-fixture.v1", "token fixture v1 required")
        requests = fixture["requests"]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        requests = [{"id": "prompt-0", "input_ids": tokenizer.encode(args.prompt, add_special_tokens=False)}]
        identity["prompt"] = args.prompt
        identity["tokenizer_files"] = {p.name: file_hash(p) for p in args.checkpoint.iterdir()
                                        if p.is_file() and ("token" in p.name or p.name in {"vocab.json", "merges.txt"})}
    result = run_loaded(model, requests, args.output, identity=identity,
                        max_new_tokens=args.max_new_tokens, updates=args.updates, learning_rate=args.learning_rate)
    return {key: value for key, value in result.items() if key not in {"divergence", "requests"}} | {
        "report": str(args.output / "result.json")}

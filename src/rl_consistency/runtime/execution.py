"""A slow, trainable common path whose physical shape depends only on one prefix.

No native batching, caches, sharding, inference-only dispatch or remote kernels.
All phases execute fresh autograd-enabled forwards; only actor retains gradients.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os

import torch

from ..common import ContractError, canonical, require


def tensor_bytes(tensor):
    value = tensor.detach().contiguous().cpu()
    require(value.numel() > 0, "empty runtime tensor")
    require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "non-finite runtime tensor")
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_record(tensor):
    raw = tensor_bytes(tensor)
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(), "data_hex": raw.hex()}


def state_hash(model):
    """Bounded host transfers; never materialize full model weights as Python lists."""
    h = hashlib.sha256()
    for role, items in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in sorted(items):
            h.update(canonical([role, name, str(value.dtype), list(value.shape)]).encode())
            flat = value.detach().reshape(-1)
            for start in range(0, flat.numel(), 1024 * 1024):
                h.update(tensor_bytes(flat[start:start + 1024 * 1024]))
    return h.hexdigest()


@contextmanager
def deterministic_execution():
    require(not torch.is_inference_mode_enabled(), "inference_mode is incompatible with actor forward")
    require(not torch.is_autocast_enabled() and not torch.is_autocast_enabled("cpu"),
            "outer autocast is unsupported; use explicit checkpoint dtype")
    if torch.version.cuda and torch.cuda.is_available():
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"},
                "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    old = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.enable_grad():
            yield
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[2:4]
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[4:]


class Divergence(ContractError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(f"bitwise mismatch: {detail['phase']} / {detail['request']} / token {detail['step']} / {detail['field']}")


@dataclass
class Trajectory:
    request: str
    prompt: list[int]
    tokens: list[int]
    observations: list[dict]
    state_sha256: str


class SharedPrefixRuntime:
    """Sequential, single-device reference executor. Use one instance from one thread.

    `generate` retains actual rollout observations. `check` independently recomputes
    them and performs real actor backward; it fails on any differing original byte.
    """

    def __init__(self, model):
        from .models import ARCHITECTURES, validate_provider_environment
        validate_provider_environment(model.config.model_type)
        adapter = ARCHITECTURES[model.config.model_type]
        self.model = model
        params = list(model.parameters())
        require(params and any(p.requires_grad for p in params), "a trainable model is required")
        devices = {p.device for p in params} | {b.device for b in model.buffers()}
        require(len(devices) == 1 and params[0].device.type in {"cpu", "cuda"}, "one CPU or GPU device required")
        require(all(p.dtype in {torch.float32, torch.bfloat16} for p in params), "unsupported parameter dtype")
        self.device = params[0].device
        self.vocab = model.config.vocab_size
        self.limit = model.config.max_position_embeddings
        self.layers = list(model.get_submodule(adapter.decoder_path))
        require(self.layers, "decoder layers required")
        self.routers = [(name, module) for name, module in model.named_modules()
                        if type(module).__name__ == adapter.router_class]
        if adapter.router_class:
            expected = sum(type(module).__name__ == adapter.sparse_layer_class for module in model.modules())
            require(expected > 0 and len(self.routers) == expected, "MoE router capture is incomplete")

    def _inputs(self, prefix):
        require(isinstance(prefix, list) and 0 < len(prefix) <= self.limit, "invalid/overlong token prefix")
        require(all(type(t) is int and 0 <= t < self.vocab for t in prefix), "token outside vocabulary")
        return torch.tensor([prefix], dtype=torch.long, device=self.device)

    def _forward(self, prefix):
        require(all(not m.training for m in self.model.modules()), "runtime requires eval mode (gradients remain enabled)")
        require(self.model.config._attn_implementation == "eager", "only eager attention is supported")
        require(not getattr(self.model, "is_gradient_checkpointing", False), "gradient checkpointing is unsupported")
        require(not getattr(self.model, "_use_kernels", False), "external kernel substitution is unsupported")
        if self.routers:
            require(self.model.config._experts_implementation == "eager", "only eager experts are supported")
        boundaries, routing, handles = [], {}, []

        def layer_hook(index):
            def capture(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                raw = tensor_bytes(value)
                boundaries.append({"layer": index, "dtype": str(value.dtype), "shape": list(value.shape),
                                   "sha256": hashlib.sha256(raw).hexdigest()})
            return capture

        def router_hook(name):
            def capture(_module, _args, output):
                require(name not in routing, "router executed more than once per prefix")
                logits, weights, experts = output
                routing[name] = {"logits": tensor_record(logits), "weights": tensor_record(weights),
                                 "experts": tensor_record(experts)}
            return capture

        try:
            for i, layer in enumerate(self.layers):
                handles.append(layer.register_forward_hook(layer_hook(i)))
            for name, module in self.routers:
                handles.append(module.register_forward_hook(router_hook(name)))
            ids = self._inputs(prefix)
            with deterministic_execution():
                output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    position_ids=torch.arange(ids.shape[1], device=self.device)[None, :],
                                    use_cache=False, logits_to_keep=1, return_dict=True)
            require(output.past_key_values is None, "model returned a cache despite use_cache=False")
            logits = output.logits[0, -1]
            require(logits.requires_grad and logits.numel() == self.vocab, "actor-capable vocabulary logits required")
            require(len(boundaries) == len(self.layers) and len(routing) == len(self.routers), "incomplete capture")
            return logits, {"logits": tensor_record(logits), "boundaries": boundaries, "routing": routing}
        finally:
            for handle in handles:
                handle.remove()

    def _observe(self, prefix, token=None):
        # Scoring and forward use the same grad/precision policy in every phase.
        with deterministic_execution():
            logits, observation = self._forward(prefix)
            token = int(logits.detach().argmax()) if token is None else token
            require(type(token) is int and 0 <= token < self.vocab, "invalid response token")
            values = logits.float()
            logprob = values[token] - torch.logsumexp(values, dim=-1)
            observation["logprob"] = tensor_record(logprob)
        return token, logprob, observation

    def generate(self, requests, *, max_new_tokens=2):
        require(type(max_new_tokens) is int and max_new_tokens > 0, "positive max_new_tokens required")
        require(isinstance(requests, list) and requests, "nonempty requests required")
        ids = [r["id"] for r in requests]
        require(all(isinstance(x, str) and x for x in ids) and len(set(ids)) == len(ids), "unique request ids required")
        identity = state_hash(self.model)
        trajectories = []
        for request in requests:
            prompt = list(request["input_ids"])
            self._inputs(prompt)
            require(len(prompt) + max_new_tokens - 1 <= self.limit, "generation exceeds context limit")
            tokens, observations = [], []
            for _ in range(max_new_tokens):
                token, logprob, observation = self._observe(prompt + tokens)
                tokens.append(token)
                observations.append(observation)
                del logprob  # Rollout is a fresh grad-enabled forward; its graph is not reused by actor.
            trajectories.append(Trajectory(request["id"], prompt, tokens, observations, identity))
        require(state_hash(self.model) == identity, "model state mutated during rollout")
        return trajectories

    def check(self, trajectories):
        require(trajectories and all(t.tokens and len(t.tokens) == len(t.observations) for t in trajectories),
                "complete nonempty rollout trajectories required")
        identity = state_hash(self.model)
        require(all(t.state_sha256 == identity for t in trajectories), "rollout and actor weight versions differ")
        self.model.zero_grad(set_to_none=True)
        count = sum(len(t.tokens) for t in trajectories)
        receipts = []
        try:
            # Reverse request order in recomputation to exercise independence from outer scheduling.
            for phase, ordered in (("old_logprob", list(reversed(trajectories))), ("actor", trajectories)):
                for trajectory in ordered:
                    for step, (token, expected) in enumerate(zip(trajectory.tokens, trajectory.observations)):
                        prefix = trajectory.prompt + trajectory.tokens[:step]
                        _, logprob, actual = self._observe(prefix, token)
                        for field in ("boundaries", "routing", "logits", "logprob"):
                            if expected[field] != actual[field]:
                                detail = {"phase": phase, "request": trajectory.request, "step": step,
                                          "prefix": prefix, "token": token, "field": field,
                                          "expected": expected[field], "actual": actual[field]}
                                if field == "boundaries":
                                    detail["first_observed_layer"] = next(i for i, (a, b) in
                                        enumerate(zip(expected[field], actual[field])) if a != b)
                                raise Divergence(detail)
                        receipts.append({"phase": phase, "request": trajectory.request, "step": step,
                                         "observation": actual})
                        if phase == "actor":
                            with deterministic_execution():
                                (-logprob / count).backward()
                        del logprob
                require(state_hash(self.model) == identity, f"model state mutated during {phase}")
            gradients, nonzero = 0, 0
            for param in self.model.parameters():
                if param.grad is not None:
                    require(bool(torch.isfinite(param.grad).all()), "non-finite actor gradient")
                    gradients += 1
                    nonzero += int(bool(torch.count_nonzero(param.grad)))
            require(nonzero > 0, "actor backward produced no nonzero gradients")
            return {"state_sha256": identity, "comparisons": receipts,
                    "parameters_with_gradient": gradients, "parameters_with_nonzero_gradient": nonzero,
                    "loss": "mean negative logprob of greedy-generated tokens; calibration only"}
        except BaseException:
            self.model.zero_grad(set_to_none=True)
            raise

"""Checkpoint adapters for the shared executor, with no RL-Kernel dependency."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from ..common import digest, file_hash, read_json, require

TRANSFORMERS_VERSION = "5.20.1"


@dataclass(frozen=True)
class Architecture:
    config_class: str
    model_class: str
    decoder_path: str = "model.layers"
    router_class: str | None = None
    sparse_layer_class: str | None = None
    excluded_packages: tuple[str, ...] = ("kernels",)


ARCHITECTURES = {
    "qwen3": Architecture("Qwen3Config", "Qwen3ForCausalLM"),
    "qwen3_moe": Architecture("Qwen3MoeConfig", "Qwen3MoeForCausalLM",
                              router_class="Qwen3MoeTopKRouter", sparse_layer_class="Qwen3MoeSparseMoeBlock"),
    "qwen3_5_text": Architecture("Qwen3_5TextConfig", "Qwen3_5ForCausalLM",
                                 excluded_packages=("kernels", "fla", "causal_conv1d")),
}
PROFILES = {
    "Qwen/Qwen3-8B": ("qwen3", {"num_hidden_layers": 32, "hidden_size": 4096,
                                "num_attention_heads": 32, "num_key_value_heads": 8,
                                "head_dim": 128, "intermediate_size": 12288, "vocab_size": 151936}),
    "Qwen/Qwen3.8-27B": ("qwen3_5_text", None),
    "Qwen/Qwen3-30B-A3B": ("qwen3_moe", None),
}


def catalog():
    return [{"model": name, "family": family, "runtime": "shared-prefix-v1",
             "rl_kernel_required": False, "architecture_cpu_tested": True,
             "full_model_gpu_verified": False, "certified": False}
            for name, (family, _) in PROFILES.items()]


def validate_provider_environment(family):
    # Transformers' Hybrid decorators can select installed FLA/conv implementations
    # even when use_kernels=False. Fail before loading rather than change global functions.
    require(family in ARCHITECTURES, "unsupported model family; register an architecture adapter")
    present = [name for name in ARCHITECTURES[family].excluded_packages if find_spec(name) is not None]
    require(not present, "shared-prefix-v1 requires an isolated reference environment without: "
            + ", ".join(present))


def validate_profile(name, config):
    require(name in PROFILES, "unregistered model; add a model profile and architecture tests first")
    family, dimensions = PROFILES[name]
    if family == "qwen3_5_text":
        from ..models.qwen38 import Qwen38Adapter
        Qwen38Adapter(config)
    elif family == "qwen3_moe":
        from ..models.qwen3_moe import Qwen3MoeAdapter
        Qwen3MoeAdapter(config)
    else:
        require(config.get("model_type") == family, "checkpoint architecture differs from model profile")
        for key, expected in dimensions.items():
            require(config.get(key) == expected, f"model profile requires {key}={expected}")
    return family


def load_checkpoint(checkpoint, family, *, device="cpu", dtype="float32"):
    """Load real local safetensors; also usable for explicitly labeled tiny test models."""
    validate_provider_environment(family)
    import torch
    import transformers
    require(transformers.__version__ == TRANSFORMERS_VERSION,
            f"runtime requires transformers=={TRANSFORMERS_VERSION}; revalidate before upgrading")
    require(dtype in {"float32", "bfloat16"}, "runtime supports float32 or bfloat16")
    root = Path(checkpoint).resolve()
    require(root.is_dir(), "checkpoint must be an existing local directory")
    raw = read_json(root / "config.json")
    text = raw.get("text_config", raw)
    require(text.get("model_type", raw.get("model_type")) == family, "checkpoint family mismatch")
    require(not raw.get("quantization_config") and not text.get("quantization_config"),
            "quantized checkpoints are not supported by this runtime")
    require((text.get("rope_parameters") or {}).get("rope_type", "default") == "default"
            and not text.get("rope_scaling"), "dynamic/scaled RoPE requires a separate runtime contract")
    # Resolve exactly the weight files used by from_pretrained, not arbitrary files in the directory.
    if (root / "model.safetensors").is_file():
        weight_paths = [root / "model.safetensors"]
    else:
        index = read_json(root / "model.safetensors.index.json")
        names = sorted(set(index["weight_map"].values()))
        require(names and all(Path(n).name == n and n.endswith(".safetensors") for n in names),
                "invalid safetensors shard paths")
        weight_paths = [root / n for n in names]
    paths = [root / "config.json", *weight_paths]
    if (root / "model.safetensors.index.json").exists():
        paths.append(root / "model.safetensors.index.json")
    checkpoint_files = {p.name: file_hash(p) for p in paths}
    adapter = ARCHITECTURES[family]
    config_class, model_class = getattr(transformers, adapter.config_class), getattr(transformers, adapter.model_class)
    config = config_class.from_pretrained(root, local_files_only=True)
    config.use_cache = False
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    model, info = model_class.from_pretrained(
        root, config=config, local_files_only=True, use_safetensors=True,
        dtype=getattr(torch, dtype), attn_implementation="eager", experts_implementation="eager",
        use_kernels=False, output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        require(not info.get(key), f"incomplete checkpoint load ({key}): {info.get(key)}")
    require(checkpoint_files == {p.name: file_hash(p) for p in paths}, "checkpoint changed during loading")
    model.gradient_checkpointing_disable()
    model.eval()
    # Preserve provider-declared FP32 parameters (e.g. Hybrid state parameters).
    model.to(device=torch.device(device))
    source_files = {"model": inspect.getfile(model_class), "torch": torch.__file__,
                    "transformers": transformers.__file__}
    source_files["torch_native"] = torch._C.__file__
    identity = {"checkpoint_files": checkpoint_files, "checkpoint_sha256": digest(checkpoint_files),
                "family": family, "model_class": model_class.__name__, "torch": torch.__version__,
                "transformers": transformers.__version__, "hip": torch.version.hip,
                "cuda": torch.version.cuda, "requested_dtype": dtype, "device": str(next(model.parameters()).device),
                "sources": {name: {"path": path, "sha256": file_hash(path)}
                            for name, path in source_files.items()}}
    if next(model.parameters()).is_cuda:
        identity["device_name"] = torch.cuda.get_device_name(next(model.parameters()).device)
    return model, identity

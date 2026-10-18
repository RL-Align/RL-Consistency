"""Qwen3.8-27B text-path inventory and semantics-preserving proxy specification.

Inspected against Qwen's config and vime d7e40fe8. No GPU/framework imports,
weight loading, kernel replacement or certification happen in this module.
"""
from __future__ import annotations

import copy
from ..common import digest, is_hash, require

MODEL_ID = "Qwen/Qwen3.8-27B"
FAMILY = "qwen3_5_hybrid_dense"
VIME_COMMIT = "d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36"
PATTERN = ["linear_attention"] * 3 + ["full_attention"]
DIMENSIONS = {
    "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408,
    "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
    "vocab_size": 248320, "linear_conv_kernel_dim": 4,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128,
}
NUMERICS = {
    "model_type": "qwen3_5_text", "dtype": "bfloat16", "hidden_act": "silu",
    "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True,
    "output_gate_type": "swish", "rms_norm_eps": 1e-6,
    "partial_rotary_factor": 0.25, "mamba_ssm_dtype": "float32",
    "tie_word_embeddings": False, "full_attention_interval": 4,
}
ROPE = {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}
COMMON_OPS = ["rms_norm_1p", "residual", "dense_swiglu"]
FULL_OPS = ["full_qkv_proj", "qk_norm", "partial_rope", "full_attention",
            "attention_output_gate", "full_o_proj"]
LINEAR_OPS = ["gdn_qkvz_proj", "gdn_ab_gates", "causal_conv1d", "gdn_qk_l2norm",
              "gated_delta_rule", "gated_rms_norm", "gdn_out_proj"]
OPERATORS = ["embedding", *COMMON_OPS, *FULL_OPS, *LINEAR_OPS,
             "lm_head", "logprob", "ordered_collectives"]
HYBRID_CASES = {
    "C3": ["gdn-chunk-recurrent", "conv-state-reset", "gdn-state-reset",
           "packed-boundaries", "cache-reorder", "prefix-cache"],
    "C7": ["gdn-forward-accuracy", "gdn-gradient-accuracy", "conv-gradient-accuracy",
           "gated-norm-gradient-accuracy"],
    "C9": ["gdn-long-context", "state-rebuild-after-sync"],
}
SCOPE = {"modality": "text-only", "weight_dtype": "bfloat16",
         "quantization": "none", "speculative_decoding": False,
         "mtp_training": False, "vision_training": False}


def is_hybrid_recipe(recipe):
    model = recipe.get("model", {})
    return model.get("id") == MODEL_ID or model.get("family") == FAMILY


def validate_hybrid_recipe(recipe, *, bound=False):
    model = recipe["model"]
    require(model.get("id") == MODEL_ID and model.get("family") == FAMILY
            and model.get("full_layers") == 64, "Qwen3.8-27B model identity differs")
    require(model.get("execution_scope") == SCOPE, "Qwen3.8 adapter currently covers BF16 text-only without MTP")
    require(set(recipe["operators"]) == set(OPERATORS), "Hybrid operator inventory incomplete")
    for gate, cases in HYBRID_CASES.items():
        require(set(cases) <= set(recipe["coverage"][gate]), f"{gate} misses Hybrid state/gradient cases")
    layers = model.get("proxy_source_layers", [])
    validate_selection(layers)
    if bound:
        require(is_hash(model.get("config_canonical_sha256")), "bind model.config_canonical_sha256 from inspected config")


def validate_selection(selected):
    require(isinstance(selected, list) and all(type(i) is int for i in selected)
            and 8 <= len(selected) < 64 and len(selected) % 4 == 0
            and selected == list(range(len(selected))),
            "Hybrid proxy must retain a contiguous prefix of 8,12,...,60 layers with complete L/L/L/F cycles")


class Qwen38Adapter:
    def __init__(self, config):
        require(isinstance(config, dict), "HF config must be an object")
        require(config.get("model_type") == "qwen3_5"
                and config.get("architectures") == ["Qwen3_5ForConditionalGeneration"],
                "expected Qwen3.8-27B nested qwen3_5 configuration")
        text = config.get("text_config")
        require(isinstance(text, dict), "nested text_config required")
        for key, expected in {**DIMENSIONS, **NUMERICS}.items():
            actual = text.get(key)
            valid_type = type(actual) is type(expected) if type(expected) in (bool, int) else not isinstance(actual, bool)
            require(actual == expected and valid_type,
                    f"unsupported text_config.{key}: expected {expected!r}")
        require(text.get("layer_types") == PATTERN * 16, "Hybrid layer order differs from L/L/L/F x 16")
        require(text.get("rope_parameters") == ROPE, "RoPE semantics differ; inspect a new adapter contract")
        require(config.get("tie_word_embeddings") is False, "untied embeddings required")
        require(not any(c.get("quantization_config") for c in (config, text)), "quantized checkpoints need a separate adapter")
        require(not any(text.get(k) for k in ("num_experts", "num_local_experts", "num_experts_per_tok")),
                "this adapter is Hybrid Dense, not MoE")
        require(type(text.get("max_position_embeddings")) is int and text["max_position_embeddings"] > 0,
                "context limit missing")
        self.config = copy.deepcopy(config)
        self.text = self.config["text_config"]

    def inventory(self):
        dimensions = {k: self.text[k] for k in DIMENSIONS if k != "num_hidden_layers"}
        layers = [{"kind": kind, "dimensions": copy.deepcopy(dimensions),
                   "precision": {"weights": "bfloat16", "recurrent_state": "float32"},
                   "operators": [*COMMON_OPS, *(LINEAR_OPS if kind == "linear_attention" else FULL_OPS)],
                   "numerics": {**NUMERICS, "rope_parameters": copy.deepcopy(ROPE)}}
                  for kind in self.text["layer_types"]]
        return {"schema": "rl-consistency.model-inventory.v1", "model_id": MODEL_ID,
                "family": FAMILY, "config_canonical_sha256": digest(self.config),
                "execution_scope": copy.deepcopy(SCOPE), "layers": layers,
                "layer_counts": {"linear_attention": 48, "full_attention": 16},
                "operators": list(OPERATORS), "certified": False,
                "identity_note": "Compatible configuration does not identify checkpoint weights or revision.",
                "state_boundaries": ["causal convolution history", "GDN recurrent matrix", "full-attention KV cache"],
                "sources": {"model_config": "https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json",
                            "vime_commit": VIME_COMMIT,
                            "vime_spec": "vime_plugins/models/qwen3_5.py",
                            "vime_model_args": "scripts/models/qwen3.5-27B.sh"}}

    def reduced_spec(self, selected_layers):
        validate_selection(selected_layers)
        inventory = self.inventory()
        reduced = copy.deepcopy(self.config)
        reduced["text_config"]["num_hidden_layers"] = len(selected_layers)
        reduced["text_config"]["layer_types"] = [self.text["layer_types"][i] for i in selected_layers]
        # Preserve every other config field, including dormant vision/MTP fields.
        # Disabling those execution paths is explicit in the execution scope.
        return {"schema": "rl-consistency.proxy-plan.v1", "execution": "not-run",
                "checkpoint_materialized": False, "certified": False,
                "source_config_canonical_sha256": digest(self.config),
                "selected_layers": list(selected_layers), "source_layers": inventory["layers"],
                "reduced_layers": [inventory["layers"][i] for i in selected_layers],
                "boundary_operators": list(OPERATORS), "config": reduced,
                "weight_layer_map": [{"source": i, "destination": j} for j,i in enumerate(selected_layers)],
                "weight_policy": "Copy original selected language layers, embedding, final norm and LM head without casting or reinitializing; audit actual tensor names and hashes. This plan contains no weights."}

    def vime_model_args(self, layers=64):
        require(layers == 64 or (type(layers) is int and 8 <= layers < 64 and layers % 4 == 0), "invalid model depth")
        # Model arguments only, matching the inspected upstream model script.
        # No GPU allocation, checkpoints, workload, optimizer or provider is selected.
        return ["--spec", "vime_plugins.models.qwen3_5", "get_qwen3_5_spec",
                "--disable-bias-linear", "--qk-layernorm", "--group-query-attention",
                "--num-attention-heads", "24", "--num-query-groups", "4", "--kv-channels", "256",
                "--num-layers", str(layers), "--hidden-size", "5120", "--ffn-hidden-size", "17408",
                "--use-gated-attention", "--normalization", "RMSNorm", "--apply-layernorm-1p",
                "--position-embedding-type", "rope", "--norm-epsilon", "1e-6", "--rotary-percent", "0.25",
                "--swiglu", "--untie-embeddings-and-output-weights", "--vocab-size", "248320",
                "--rotary-base", "10000000", "--attention-output-gate"]


def check_model_evidence(gate, evidence, recipe, artifact):
    """Check actual config artifacts, rather than accepting claimed layer counts."""
    if gate not in {"C0", "C2", "C9"}:
        return
    source = artifact(evidence["model_config"])
    adapter = Qwen38Adapter(source)
    require(digest(source) == recipe["model"]["config_canonical_sha256"], "Hybrid source config identity differs")
    if gate == "C2":
        expected = adapter.reduced_spec(recipe["model"]["proxy_source_layers"])
        coverage = artifact(evidence["coverage"])
        for key in ("selected_layers", "source_layers", "reduced_layers", "boundary_operators"):
            require(coverage.get(key) == expected[key], f"Hybrid proxy {key} differs from actual source config")
        require(artifact(evidence["proxy_config"]) == expected["config"],
                "Hybrid proxy changed non-depth config fields or layer order")
        require(coverage.get("checkpoint_materialized") is True
                and is_hash(coverage.get("weight_manifest_sha256")), "materialized proxy weight manifest required")

"""Qwen3-30B-A3B BF16 onboarding; no kernels, weights or GPU certification."""
from __future__ import annotations

import copy
from ..common import digest, is_hash, require

MODEL_ID = "Qwen/Qwen3-30B-A3B"
FAMILY = "qwen3_moe"
VIME_COMMIT = "d7e40fe8ed9fd9ab61ecb0ec2259b3cc4568af36"
DIMENSIONS = {
    "num_hidden_layers": 48, "hidden_size": 2048, "intermediate_size": 6144,
    "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
    "vocab_size": 151936, "moe_intermediate_size": 768,
    "num_experts": 128, "num_experts_per_tok": 8,
}
NUMERICS = {
    "model_type": "qwen3_moe", "torch_dtype": "bfloat16", "hidden_act": "silu",
    "attention_bias": False, "attention_dropout": 0.0, "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0, "rope_scaling": None, "tie_word_embeddings": False,
    "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": True,
    "use_sliding_window": False, "sliding_window": None,
}
LAYER_OPS = ["rms_norm", "residual", "qkv_proj", "qk_norm", "rope", "full_attention",
             "o_proj", "router_gemm", "router_softmax", "router_topk", "router_normalize",
             "moe_dispatch", "expert_swiglu", "moe_combine", "ordered_collectives"]
OPERATORS = ["embedding", *LAYER_OPS, "lm_head", "logprob"]
SCOPE = {"modality": "text-only", "weight_dtype": "bfloat16", "quantization": "none",
         "speculative_decoding": False, "routing_replay": False}
# Intended strict contract, not a claim about unmodified framework kernels.
ROUTING = {"num_experts": 128, "top_k": 8, "score_function": "softmax",
           "normalize_topk": True, "router_dtype": "float32",
           "weight_stage": "normalized_pre_cast", "weight_dtype": "float32",
           "tie_break": "lower_expert_id", "combine_order": "ascending_slot",
           "drop_tokens": False, "independent_routing": True}
MOE_CASES = {
    "C3": ["router-ties", "expert-imbalance", "dispatch-permutation", "empty-expert", "expert-capacity"],
    "C7": ["router-gradient-accuracy", "expert-gradient-accuracy", "combine-gradient-accuracy"],
    "C9": ["expert-parallel", "routing-after-sync"],
}


def validate_selection(selected):
    require(isinstance(selected, list) and all(type(i) is int for i in selected)
            and 4 <= len(selected) < 48 and selected == list(range(len(selected))),
            "MoE proxy must retain a contiguous prefix of 4..47 complete layers")


def validate_moe_recipe(recipe, *, bound=False):
    model = recipe["model"]
    require(model.get("id") == MODEL_ID and model.get("family") == FAMILY
            and model.get("full_layers") == 48, "Qwen3-30B-A3B model identity differs")
    require(model.get("execution_scope") == SCOPE, "MoE adapter requires BF16 text and independent routing")
    require(recipe.get("routing") == ROUTING, "MoE strict routing contract differs")
    require(set(recipe["operators"]) == set(OPERATORS), "MoE operator inventory incomplete")
    for gate, cases in MOE_CASES.items():
        require(set(cases) <= set(recipe["coverage"][gate]), f"{gate} misses MoE routing/gradient cases")
    validate_selection(model.get("proxy_source_layers"))
    for side in ("training", "rollout"):
        require(type(recipe["target"][side].get("expert_tp")) is int
                and recipe["target"][side]["expert_tp"] > 0, f"bind {side}.expert_tp explicitly")
    require(recipe.get("moe_training") == {"aux_loss_coef": 0.0, "token_capacity_factor": None},
            "first MoE contract uses no auxiliary routing loss and no token capacity limit")
    if bound:
        require(is_hash(model.get("config_canonical_sha256")), "bind model.config_canonical_sha256 from inspected config")


class Qwen3MoeAdapter:
    model_id = MODEL_ID
    recipe_prefix = "qwen3-30b-a3b-vime"
    default_proxy_layers = 4
    full_layers = 48
    vime_commit = VIME_COMMIT
    guide = "qwen3-30b-a3b.md"
    blockers = ["MoE strict router, dispatch, expert, combine and collective providers",
                "vLLM/Megatron dual-path and independently computed routing capture"]

    @staticmethod
    def validate_depth(layers):
        require(type(layers) is int and 4 <= layers < 48, "MoE proxy layer count must be 4..47")

    def __init__(self, config):
        require(isinstance(config, dict), "HF config must be an object")
        require(config.get("architectures") == ["Qwen3MoeForCausalLM"], "expected Qwen3MoeForCausalLM config")
        for key, expected in {**DIMENSIONS, **NUMERICS}.items():
            actual = config.get(key)
            valid_type = type(actual) is type(expected) if type(expected) in (bool, int) else not isinstance(actual, bool)
            require(key in config and actual == expected and valid_type, f"unsupported {key}: expected {expected!r}")
        require(config.get("dtype", "bfloat16") == "bfloat16", "conflicting checkpoint dtype")
        require(not config.get("quantization_config"), "quantized checkpoints need a separate adapter")
        require(not any(config.get(k) for k in ("num_shared_experts", "n_shared_experts", "shared_expert_intermediate_size")),
                "Qwen3-30B-A3B has no shared experts")
        require(type(config.get("max_position_embeddings")) is int and config["max_position_embeddings"] > 0,
                "context limit missing")
        self.config = copy.deepcopy(config)

    def inventory(self):
        layer = {"kind": "full_attention_moe",
                 "dimensions": {k: self.config[k] for k in DIMENSIONS if k != "num_hidden_layers"},
                 "precision": {"weights": "bfloat16", "router": "float32"},
                 "operators": list(LAYER_OPS), "numerics": copy.deepcopy(NUMERICS),
                 "routing_contract": copy.deepcopy(ROUTING)}
        return {"schema": "rl-consistency.model-inventory.v1", "model_id": MODEL_ID,
                "family": FAMILY, "config_canonical_sha256": digest(self.config),
                "execution_scope": copy.deepcopy(SCOPE), "layers": [copy.deepcopy(layer) for _ in range(48)],
                "layer_counts": {"full_attention_moe": 48}, "operators": list(OPERATORS), "certified": False,
                "identity_note": "Compatible configuration does not identify checkpoint weights or revision.",
                "state_boundaries": ["full-attention KV cache", "token/expert/slot dispatch map", "expert process groups"],
                "sources": {"model_config": "https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json",
                            "vime_commit": VIME_COMMIT, "vime_model_args": "scripts/models/qwen3-30B-A3B.sh"}}

    def reduced_spec(self, selected_layers):
        validate_selection(selected_layers)
        inventory = self.inventory()
        reduced = copy.deepcopy(self.config)
        reduced["num_hidden_layers"] = len(selected_layers)
        return {"schema": "rl-consistency.proxy-plan.v1", "execution": "not-run",
                "checkpoint_materialized": False, "certified": False,
                "source_config_canonical_sha256": digest(self.config),
                "selected_layers": list(selected_layers), "source_layers": inventory["layers"],
                "reduced_layers": [inventory["layers"][i] for i in selected_layers],
                "boundary_operators": list(OPERATORS), "config": reduced,
                "weight_layer_map": [{"source": i, "destination": j} for j, i in enumerate(selected_layers)],
                "weight_policy": "Copy selected full layers including all 128 experts and routers, embedding, final norm and LM head without casting or reinitializing; audit tensor names and hashes. No weights are materialized."}

    def vime_model_args(self, layers=48):
        require(type(layers) is int and 4 <= layers <= 48, "invalid model depth")
        return ["--disable-bias-linear", "--qk-layernorm", "--group-query-attention",
                "--num-attention-heads", "32", "--num-query-groups", "4", "--kv-channels", "128",
                "--num-layers", str(layers), "--hidden-size", "2048", "--ffn-hidden-size", "6144",
                "--normalization", "RMSNorm", "--position-embedding-type", "rope", "--norm-epsilon", "1e-6",
                "--rotary-percent", "1.0", "--swiglu", "--untie-embeddings-and-output-weights",
                "--vocab-size", "151936", "--rotary-base", "1000000", "--moe-ffn-hidden-size", "768",
                "--moe-router-score-function", "softmax", "--moe-token-dispatcher-type", "alltoall",
                "--moe-router-topk", "8", "--moe-layer-freq", "[" + ",".join(["1"] * layers) + "]",
                "--num-experts", "128", "--moe-grouped-gemm", "--moe-token-drop-policy", "probs",
                "--moe-router-dtype", "fp32", "--moe-permute-fusion", "--moe-aux-loss-coeff", "0"]


def check_model_evidence(gate, evidence, recipe, artifact):
    if gate not in {"C0", "C2", "C9"}:
        return
    source = artifact(evidence["model_config"])
    adapter = Qwen3MoeAdapter(source)
    require(digest(source) == recipe["model"]["config_canonical_sha256"], "MoE source config identity differs")
    if gate == "C2":
        expected = adapter.reduced_spec(recipe["model"]["proxy_source_layers"])
        coverage = artifact(evidence["coverage"])
        for key in ("selected_layers", "source_layers", "reduced_layers", "boundary_operators"):
            require(coverage.get(key) == expected[key], f"MoE proxy {key} differs from actual source config")
        require(artifact(evidence["proxy_config"]) == expected["config"], "MoE proxy changed non-depth config fields")
        require(coverage.get("checkpoint_materialized") is True and is_hash(coverage.get("weight_manifest_sha256")),
                "materialized proxy weight manifest required")

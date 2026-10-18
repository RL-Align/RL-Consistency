"""Model preparation adapters; availability is not numerical certification."""
from . import qwen38, qwen3_moe
from ..common import require

ADAPTERS = {qwen38.MODEL_ID: qwen38.Qwen38Adapter,
            qwen3_moe.MODEL_ID: qwen3_moe.Qwen3MoeAdapter}


def get_adapter(model_id, config):
    require(model_id in ADAPTERS, "no preparation adapter for this model; Qwen3-8B uses its existing bridge")
    return ADAPTERS[model_id](config)


def recipe_module(recipe):
    model = recipe.get("model", {})
    matches = [module for module in (qwen38, qwen3_moe)
               if model.get("id") == module.MODEL_ID or model.get("family") == module.FAMILY]
    require(len(matches) <= 1, "conflicting model identity and family")
    return matches[0] if matches else None


def validate_model_recipe(recipe, *, bound=False):
    module = recipe_module(recipe)
    if module is qwen38:
        module.validate_hybrid_recipe(recipe, bound=bound)
    elif module is not None:
        module.validate_moe_recipe(recipe, bound=bound)


def check_model_evidence(gate, evidence, recipe, artifact):
    module = recipe_module(recipe)
    if module is not None:
        module.check_model_evidence(gate, evidence, recipe, artifact)

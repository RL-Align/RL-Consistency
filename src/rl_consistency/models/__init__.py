"""Model preparation adapters; availability is not numerical certification."""
from .qwen38 import MODEL_ID, Qwen38Adapter
from ..common import require


def get_adapter(model_id, config):
    require(model_id == MODEL_ID, "no preparation adapter for this model; Qwen3-8B uses its existing bridge")
    return Qwen38Adapter(config)

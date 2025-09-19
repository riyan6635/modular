# /max/pipelines/architectures/gemma3multimodal/weight_adapters.py

from __future__ import annotations
from typing import Dict, Any

# Correct imports from the actual modular repository
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

# Maps from Safetensor to MAX weight names.
GEMMA3_SAFETENSOR_MAP: dict[str, str] = {
    "language_model.model.": "language_model.",
    # Add your custom mappings here for vision components
    "vision_tower.vision_model.embeddings.": "vision_tower.",
    "multi_modal_projector.": "multi_modal_projector.",
}

def _apply_name_mappings(name: str) -> str:
    """Apply all name mappings to a given name."""
    for before, after in GEMMA3_SAFETENSOR_MAP.items():
        name = name.replace(before, after)
    return name

def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format"""
    new_state_dict: dict[str, WeightData] = {}

    # Remap HuggingFace -> MAX-style names
    for weight_name, value in state_dict.items():
        max_name = _apply_name_mappings(weight_name)
        new_state_dict[max_name] = value.data()

    # For quantized model, apply same name re-mapping to the `ignore` list
    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    if hf_quant_config and "ignore" in hf_quant_config:
        hf_quant_config["ignore"] = [
            _apply_name_mappings(module_name)
            for module_name in hf_quant_config["ignore"]
        ]

    return new_state_dict

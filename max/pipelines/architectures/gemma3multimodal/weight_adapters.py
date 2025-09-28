from __future__ import annotations
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """
    Convert Gemma3 Multimodal SafeTensor weights to MAX format.
    Handles text model, vision tower, and multimodal projector weights.
    """
    new_state_dict: dict[str, WeightData] = {}
    
    print(f"Converting {len(state_dict)} weights for Gemma3 Multimodal")
    
    for weight_name, value in state_dict.items():
        original_name = weight_name
        max_name = weight_name
        
        # Handle text model weights - these need language_model prefix
        if weight_name == "model.embed_tokens.weight":
            max_name = "language_model.embed_tokens.weight"
        elif weight_name.startswith("model.layers."):
            max_name = weight_name.replace("model.", "language_model.")
        elif weight_name.startswith("model.") and not weight_name.startswith("language_model."):
            # For other model.* weights, add language_model prefix
            max_name = weight_name.replace("model.", "language_model.")
        elif weight_name.startswith("language_model.model."):
            # Remove redundant .model. part
            max_name = weight_name.replace("language_model.model.", "language_model.")
        
        # Handle vision tower weights - keep as-is, these are correct
        elif weight_name.startswith("vision_tower."):
            max_name = weight_name
            
        # Handle multimodal projector weights - keep as-is
        elif weight_name.startswith("multi_modal_projector."):
            max_name = weight_name
            
        # Handle other special cases
        elif weight_name == "lm_head.weight":
            max_name = "language_model.lm_head.weight"
            
        # Store the mapped weight
        try:
            new_state_dict[max_name] = value.data()
        except Exception as e:
            print(f"Failed to convert weight {weight_name}: {e}")
            continue
        
        # Log mapping if changed
        if weight_name != max_name:
            print(f"Weight mapping: {weight_name} -> {max_name}")
    
    # Handle quantization config if present
    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    if hf_quant_config and "ignore" in hf_quant_config:
        updated_ignore = []
        for module_name in hf_quant_config["ignore"]:
            if module_name == "model.embed_tokens.weight":
                updated_ignore.append("language_model.embed_tokens.weight")
            elif module_name.startswith("model.") and not module_name.startswith("language_model."):
                updated_ignore.append(module_name.replace("model.", "language_model."))
            elif module_name.startswith("language_model.model."):
                updated_ignore.append(module_name.replace("language_model.model.", "language_model."))
            else:
                updated_ignore.append(module_name)
        hf_quant_config["ignore"] = updated_ignore
    
    # Log weight categories
    vision_weights = sum(1 for name in new_state_dict.keys() if name.startswith("vision_tower."))
    projector_weights = sum(1 for name in new_state_dict.keys() if name.startswith("multi_modal_projector."))
    language_weights = sum(1 for name in new_state_dict.keys() if name.startswith("language_model."))
    
    print(f"Successfully converted {len(state_dict)} -> {len(new_state_dict)} weights")
    print(f"Weight distribution: Language={language_weights}, Vision={vision_weights}, Projector={projector_weights}")
    
    return new_state_dict
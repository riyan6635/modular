from dataclasses import dataclass
from typing import Optional

@dataclass
class Gemma3MultiModalConfig:
    """Configuration for Gemma3 multimodal model"""
    
    # Vision configuration
    vision_config: Optional[dict] = None
    image_size: int = 896
    patch_size: int = 14
    vision_hidden_size: int = 1152
    vision_layers: int = 27
    vision_heads: int = 16
    
    # Text configuration
    vocab_size: int = 262208
    hidden_size: int = 5376
    num_hidden_layers: int = 62
    num_attention_heads: int = 32
    num_key_value_heads: int = 16
    intermediate_size: int = 21504
    max_position_embeddings: int = 131072
    
    # Multimodal configuration
    mm_projector_type: str = "average_pooling"
    mm_use_proj_bias: bool = False
    
    # Model-specific parameters
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    mlp_bias: bool = False
    
    @classmethod
    def from_huggingface_config(cls, hf_config):
        """Create config from HuggingFace model config"""
        
        # Extract vision config if present
        vision_config = getattr(hf_config, 'vision_config', None)
        if hasattr(vision_config, 'to_dict'):
            vision_config = vision_config.to_dict()
        
        return cls(
            model_name=getattr(hf_config, 'model_name', 'gemma-3-4b-it'),
            vocab_size=getattr(hf_config, 'vocab_size', 262208),
            hidden_size=getattr(hf_config, 'hidden_size', 5376),
            num_hidden_layers=getattr(hf_config, 'num_hidden_layers', 62),
            num_attention_heads=getattr(hf_config, 'num_attention_heads', 32),
            num_key_value_heads=getattr(hf_config, 'num_key_value_heads', 16),
            intermediate_size=getattr(hf_config, 'intermediate_size', 21504),
            max_position_embeddings=getattr(hf_config, 'max_position_embeddings', 131072),
            vision_config=vision_config,
            image_size=getattr(vision_config, 'image_size', 896) if vision_config else 896,
            patch_size=getattr(vision_config, 'patch_size', 14) if vision_config else 14,
            vision_hidden_size=getattr(vision_config, 'hidden_size', 1152) if vision_config else 1152,
            vision_layers=getattr(vision_config, 'num_hidden_layers', 27) if vision_config else 27,
            vision_heads=getattr(vision_config, 'num_attention_heads', 16) if vision_config else 16,
            rope_theta=getattr(hf_config, 'rope_theta', 10000.0),
            rms_norm_eps=getattr(hf_config, 'rms_norm_eps', 1e-6),
        )

from dataclasses import dataclass
from typing import Optional, List, Tuple
import math
import logging
logger = logging.getLogger(__name__)

@dataclass
class SigLIPVisionConfig:
    """Configuration for SigLIP vision encoder."""
    def __call__(self, pixel_values):
        logger.info(f"VisionEncoder got tensor with shape {pixel_values.shape} "
                    f"and dtype {pixel_values.dtype}; min={pixel_values.min().item()}, max={pixel_values.max().item()}")
      
    # Image processing
    image_size: int = 896  # Input image resolution (896x896)
    patch_size: int = 14   # Patch size for Vision Transformer
    num_channels: int = 3  # RGB channels
    
    # Model architecture
    hidden_size: int = 1152       # Hidden dimension
    intermediate_size: int = 4304  # MLP intermediate size
    num_hidden_layers: int = 27    # Number of transformer layers
    num_attention_heads: int = 16  # Number of attention heads
    
    # Normalization and regularization
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Activation function
    hidden_act: str = "gelu"
    
    # Positional embeddings
    use_learned_pos_embeddings: bool = True
    
    @property
    def num_patches(self) -> int:
        """Calculate number of patches."""
        return (self.image_size // self.patch_size) ** 2
    
    @property
    def head_dim(self) -> int:
        """Calculate attention head dimension."""
        return self.hidden_size // self.num_attention_heads


@dataclass 
class CrossModalProjectorConfig:
    """Configuration for cross-modal projector."""
    
    # Input/output dimensions
    vision_hidden_size: int = 1152  # From SigLIP
    language_hidden_size: int = 3072  # From Gemma3
    
    # Projector architecture
    projector_type: str = "linear"  # "linear" or "mlp"
    num_layers: int = 1
    intermediate_size: Optional[int] = None
    
    # Regularization
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    
    # Activation
    hidden_act: str = "gelu"


@dataclass
class Gemma3MultimodalConfig:
    """Complete configuration for Gemma3 Multimodal model."""
    
    # Component configurations
    vision_config: SigLIPVisionConfig
    projector_config: CrossModalProjectorConfig
    
    # Text model config (will be populated from HF config)
    text_config: Optional[dict] = None
    
    # Multimodal specific settings
    image_token_index: int = 256000  # Special token for image placeholder
    num_image_tokens: int = 256      # Number of tokens per image (64x64 patches for 896x896)
    vision_feature_layer: int = -2   # Which vision layer to extract features from
    vision_feature_select_strategy: str = "default"
    
    # Processing settings
    max_image_size: int = 896
    min_image_size: int = 224
    image_aspect_ratio: str = "pad"  # "pad", "crop", or "resize"
    
    # Training settings
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    vision_lr_multiplier: float = 1.0
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.vision_config is None:
            self.vision_config = SigLIPVisionConfig()
        if self.projector_config is None:
            self.projector_config = CrossModalProjectorConfig()
    
    def get_num_vision_tokens(self) -> int:
        """Calculate the number of vision tokens based on image size and patch size."""
        patches_per_side = self.vision_config.image_size // self.vision_config.patch_size
        return patches_per_side * patches_per_side
    
    def get_total_params(self) -> int:
        """Estimate total number of parameters."""
        # Vision encoder parameters
        vision_params = self._estimate_vision_params()
        
        # Projector parameters  
        projector_params = self._estimate_projector_params()
        
        # Note: Text model params are handled by parent Gemma3 model
        return vision_params + projector_params
    
    def _estimate_vision_params(self) -> int:
        """Estimate vision encoder parameters."""
        config = self.vision_config
        
        # Patch embedding
        patch_embed_params = (config.patch_size ** 2 * config.num_channels + 1) * config.hidden_size
        
        # Position embeddings
        pos_embed_params = (config.num_patches + 1) * config.hidden_size  # +1 for cls token
        
        # Transformer layers
        layer_params = (
            # Multi-head attention
            4 * config.hidden_size * config.hidden_size +  # qkv + output projection
            # Layer norms
            2 * config.hidden_size +
            # MLP
            2 * config.hidden_size * config.intermediate_size
        )
        transformer_params = layer_params * config.num_hidden_layers
        
        # Final layer norm
        final_norm_params = config.hidden_size
        
        return patch_embed_params + pos_embed_params + transformer_params + final_norm_params
    
    def _estimate_projector_params(self) -> int:
        """Estimate cross-modal projector parameters."""
        config = self.projector_config
        
        if config.projector_type == "linear":
            return config.vision_hidden_size * config.language_hidden_size
        elif config.projector_type == "mlp":
            intermediate_size = config.intermediate_size or (config.vision_hidden_size * 2)
            return (
                config.vision_hidden_size * intermediate_size +
                intermediate_size * config.language_hidden_size +
                intermediate_size + config.language_hidden_size  # biases
            )
        else:
            return 0
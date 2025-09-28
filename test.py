import numpy as np
import torch
from max.driver import  Device, Tensor
from max.dtype import DType
from max.graph.weights import Weights
from max.pipelines.architectures.gemma3multimodal.cross_attention import SigLIPVisionEncoder  # Adjust import path as needed
from max.pipelines.architectures.gemma3multimodal.model_config import SigLIPVisionConfig  # Adjust import path as needed

def create_mock_weights(config):
    weights = {}
    
    # Helper to create random weight tensor and wrap as MAX Tensor
    def rand_tensor(shape):
        data = np.random.uniform(-1, 1, shape).astype(np.float32)
        return Tensor.from_numpy(data).to(device)
    
    # Patch embedding weights & bias
    weights["vision_tower.vision_model.embeddings.patch_embedding.weight"] = rand_tensor(
        (config.hidden_size, config.num_channels, config.patch_size, config.patch_size)
    )
    weights["vision_tower.vision_model.embeddings.patch_embedding.bias"] = rand_tensor(
        (config.hidden_size,)
    )
    
    # Position embedding
    weights["vision_tower.vision_model.embeddings.position_embedding"] = rand_tensor(
        (config.num_patches, config.hidden_size)
    )
    
    # Encoder layers weights (for simplicity, one layer only)
    layer_idx = 0
    prefix = f"vision_tower.vision_model.encoder.layers.{layer_idx}"
    for attn_part in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        weights[f"{prefix}.self_attn.{attn_part}.weight"] = rand_tensor(
            (config.hidden_size, config.hidden_size)
        )
        weights[f"{prefix}.self_attn.{attn_part}.bias"] = rand_tensor((config.hidden_size,))
    
    # Layer norm weights
    weights[f"{prefix}.layer_norm1.weight"] = rand_tensor((config.hidden_size,))
    weights[f"{prefix}.layer_norm1.bias"] = rand_tensor((config.hidden_size,))
    weights[f"{prefix}.layer_norm2.weight"] = rand_tensor((config.hidden_size,))
    weights[f"{prefix}.layer_norm2.bias"] = rand_tensor((config.hidden_size,))
    
    # MLP weights
    weights[f"{prefix}.mlp.fc1.weight"] = rand_tensor((config.intermediate_size, config.hidden_size))
    weights[f"{prefix}.mlp.fc1.bias"] = rand_tensor((config.intermediate_size,))
    weights[f"{prefix}.mlp.fc2.weight"] = rand_tensor((config.hidden_size, config.intermediate_size))
    weights[f"{prefix}.mlp.fc2.bias"] = rand_tensor((config.hidden_size,))
    
    # Post layer norm
    weights["vision_tower.vision_model.post_layernorm.weight"] = rand_tensor((config.hidden_size,))
    weights["vision_tower.vision_model.post_layernorm.bias"] = rand_tensor((config.hidden_size,))
    
    # Wrap weights dict as MAX Weights object if needed (depends on your framework)
    # For testing just use dict or your real method
    return weights


if __name__ == "__main__":
    # Setup device
    device = GPU(0) if torch.cuda.is_available() else Device.CPU()
    print(f"Using device: {device}")
    
    # Setup config with example parameters matching your model
    config = SigLIPVisionConfig(
        num_channels=3,
        image_size=896,
        patch_size=14,
        hidden_size=1152,
        num_patches=(896 // 14) ** 2,
        num_attention_heads=16,
        num_hidden_layers=1,  # Test with 1 layer for speed
        intermediate_size=4304,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="gelu",
    )
    
    # Create or load weights dictionary
    weights = create_mock_weights(config)
    
    # Initialize the encoder
    encoder = SigLIPVisionEncoder(config, weights, DType.float32, device)
    print("SigLIPVisionEncoder initialized successfully.")
    
    # Create dummy input image batch tensor
    batch_size = 1
    # Create random tensor as dummy batch
    pixel_values = Tensor.from_numpy(np.random.uniform(0, 255, (batch_size, config.num_channels, config.image_size, config.image_size)).astype(np.float32)).to(device)
    
    # Run forward pass
    output = encoder(pixel_values)
    print(f"Encoder output shape: {output.shape}")

from __future__ import annotations
import math
from typing import Optional, Tuple
import logging

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, TensorValue, ops
from max.nn import Linear, LayerNorm
from max.nn.layer import Layer
from max.graph.weights import Weights
import time
from .model_config import SigLIPVisionConfig, CrossModalProjectorConfig
import traceback
logger = logging.getLogger(__name__)

class SigLIPPatchEmbedding(Layer):
    """Converts image patches to embeddings using Conv2D."""
    
    def __init__(
        self,
        config: SigLIPVisionConfig,
        weights: Weights,
        dtype: DType,
        device: DeviceRef,
    ):
        self.config = config
        print("Initializing SigLIP Patch Embedding")
        
        # Try multiple weight key patterns for patch embedding
        weight_keys_to_try = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_model.embeddings.patch_embedding.weight", 
            "vision_tower.embeddings.patch_embedding.weight",
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        ]
        
        bias_keys_to_try = [
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_model.embeddings.patch_embedding.bias",
            "vision_tower.embeddings.patch_embedding.bias", 
            "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
        ]
        
        self.weight = None
        self.bias = None
        
        # Try to load actual weights
        for weight_key in weight_keys_to_try:
            try:
                self.weight = weights[weight_key].allocate(
                    dtype,
                    [config.hidden_size, config.num_channels, config.patch_size, config.patch_size],
                    device=device,
                )
                print(f"Loaded patch embedding weights from: {weight_key}")
                break
            except (KeyError, AttributeError):
                continue
        
        # Try to load bias
        for bias_key in bias_keys_to_try:
            try:
                self.bias = weights[bias_key].allocate(
                    dtype,
                    [config.hidden_size],
                    device=device,
                )
                print(f"Loaded patch embedding bias from: {bias_key}")
                break
            except (KeyError, AttributeError):
                continue
        
        # If no actual weights found, create learnable parameters with proper initialization
        if self.weight is None:
            print("Patch embedding weights not found, creating initialized parameters")
            # Xavier/Glorot uniform initialization for Conv2d
            fan_in = config.num_channels * config.patch_size * config.patch_size
            fan_out = config.hidden_size * config.patch_size * config.patch_size
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            
            import numpy as np
            from max.driver import Tensor
            
            weight_data = np.random.uniform(
                -bound, bound, 
                (config.hidden_size, config.num_channels, config.patch_size, config.patch_size)
            ).astype(np.float32)
            
            self.weight = Tensor.from_numpy(weight_data).to(device)
            print("Initialized patch embedding weights with Xavier uniform")
        
        if self.bias is None:
            self.bias = ops.zeros([config.hidden_size])
    
    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        """Convert image to patch embeddings using Conv2D."""
        # Apply 2D convolution for patch embedding
        # Input: [batch, channels, height, width]
        # Output: [batch, hidden_size, num_patches_h, num_patches_w]
        embeddings = ops.conv2d(
            pixel_values,
            self.weight,
            bias=self.bias,
            stride=[self.config.patch_size, self.config.patch_size],
            padding=[0, 0]
        )
        
        # Get dimensions
        batch_size = embeddings.shape[0]
        hidden_size = embeddings.shape[1]
        height_patches = embeddings.shape[2]
        width_patches = embeddings.shape[3]
        
        # Reshape to [batch, hidden_size, num_patches]
        num_patches = height_patches * width_patches
        embeddings = embeddings.reshape([batch_size, hidden_size, num_patches])
        
        # Transpose to [batch, num_patches, hidden_size] (standard transformer format)
        embeddings = ops.transpose(embeddings, [0, 2, 1])
        
        return embeddings


class SigLIPMultiHeadAttention(Layer):    
    def __init__(self, config, weights, layer_idx, dtype, device):
        import time
        
        self.config = config
        self.layer_idx = layer_idx
        
        # ✅ Initialize attention parameters first
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.num_heads = config.num_attention_heads
        
        # ✅ Initialize dropout
        if hasattr(config, 'attention_dropout') and config.attention_dropout > 0.0:
            self.dropout = lambda x: ops.dropout(x, config.attention_dropout)
        else:
            self.dropout = None
        
        # ✅ Check cache (fixed)
        cache_key = f"{layer_idx}_{config.hidden_size}_{config.num_attention_heads}"
        if not hasattr(self.__class__, '_weight_key_cache'):
            self.__class__._weight_key_cache = {}
            
        if cache_key in self.__class__._weight_key_cache:
            base_key = self.__class__._weight_key_cache[cache_key]
            print(f"[LAYER {layer_idx}] Using cached weight key: {base_key}")
            try:
                self._load_weights(base_key, weights, config, dtype, device)
                return
            except Exception as e:
                print(f"[LAYER {layer_idx}] Cached key failed: {e}")
                # Fall through to try other keys

        base_keys_to_try = [
            f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn",
            f"vision_model.encoder.layers.{layer_idx}.self_attn",
            f"vision_tower.encoder.layers.{layer_idx}.self_attn",
            f"model.vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn",
        ]
        base_keys = [f"vision_tower.vision_model.encoder.layers.{name}.weight" for name in ["q_proj", "k_proj", "v_proj", "out_proj"]]
        
        success = False
        for base_key in base_keys:
            print(f"[LAYER {layer_idx}] Trying weight key: {base_key}")
            try:
                self._load_weights(base_key, weights, config, dtype, device)
                print(f"[LAYER {layer_idx}] SUCCESS with {base_key}")
                self.__class__._weight_key_cache[cache_key] = base_key
                success = True
                break
            except Exception as e:
                print(f"[LAYER {layer_idx}] FAILED {base_key}: {e}")
                continue
        
        if not success:
            print(f"[LAYER {layer_idx}] Creating initialized attention projections")
            self._create_initialized_projections(config, dtype, device)

    def _load_weights(self, base_key, weights, config, dtype, device):
        import time
        
        # ✅ Validate all keys exist first
        required_keys = [f"{base_key}.{name}.weight" for name in ["q_proj", "k_proj", "v_proj", "out_proj"]]
        missing_keys = [key for key in required_keys if key not in weights]
        if missing_keys:
            raise KeyError(f"Missing weights: {missing_keys}")
        
        # ✅ Load weights with proper error handling
        def safe_alloc_weight(name, shape):
            start = time.time()
            try:
                w = weights[f"{base_key}.{name}.weight"].allocate(dtype, shape, device=device)
                elapsed = time.time() - start
                print(f"[LAYER {self.layer_idx}] {name} weight allocated in {elapsed:.2f}s")
                return w
            except Exception as e:
                elapsed = time.time() - start
                print(f"[LAYER {self.layer_idx}] {name} weight FAILED in {elapsed:.2f}s: {e}")
                raise
        
        def safe_alloc_bias(name, shape):
            bias_key = f"{base_key}.{name}.bias"
            if bias_key in weights:
                start = time.time()
                try:
                    b = weights[bias_key].allocate(dtype, shape, device=device)
                    elapsed = time.time() - start
                    print(f"[LAYER {self.layer_idx}] {name} bias allocated in {elapsed:.2f}s")
                    return b
                except Exception as e:
                    elapsed = time.time() - start
                    print(f"[LAYER {self.layer_idx}] {name} bias FAILED in {elapsed:.2f}s: {e}")
                    return None
            return None
        
        # Create projections
        from torch import nn

        self.q_proj = nn.Linear(
            safe_alloc_weight("q_proj", [config.hidden_size, config.hidden_size]), 
            safe_alloc_bias("q_proj", [config.hidden_size])
        )
        self.k_proj = nn.Linear(
            safe_alloc_weight("k_proj", [config.hidden_size, config.hidden_size]), 
            safe_alloc_bias("k_proj", [config.hidden_size])
        )
        self.v_proj = nn.Linear(
            safe_alloc_weight("v_proj", [config.hidden_size, config.hidden_size]), 
            safe_alloc_bias("v_proj", [config.hidden_size])
        )
        self.out_proj = nn.Linear(
            safe_alloc_weight("out_proj", [config.hidden_size, config.hidden_size]), 
            safe_alloc_bias("out_proj", [config.hidden_size])
        )
    
    def _create_initialized_projections(self, config, dtype, device):
        """Create properly initialized attention projections."""
        import numpy as np
        from max.driver import Tensor
        
        # Xavier uniform initialization
        bound = math.sqrt(6.0 / (2 * config.hidden_size))
        
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            weight_data = np.random.uniform(
                -bound, bound, (config.hidden_size, config.hidden_size)
            ).astype(np.float32)
            weight = Tensor.from_numpy(weight_data).to(device)
            
            projection = Linear(weight, None)
            setattr(self, proj_name, projection)
    
    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Apply multi-head self-attention."""
        batch_size, seq_len, hidden_size = (
            hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]
        )
        
        # Linear projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states) 
        value = self.v_proj(hidden_states)
        
        # Reshape to multi-head format: [batch, seq_len, num_heads, head_dim]
        query = query.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        key = key.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        value = value.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose to [batch, num_heads, seq_len, head_dim] for efficient attention
        query = ops.transpose(query, [0, 2, 1, 3])
        key = ops.transpose(key, [0, 2, 1, 3])
        value = ops.transpose(value, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scores = ops.matmul(query, ops.transpose(key, [0, 1, 3, 2])) * self.scale
        
        # Apply softmax
        attn_weights = ops.softmax(scores, axis=-1)
        
        # Apply dropout if specified
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = ops.matmul(attn_weights, value)
        
        # Transpose back to [batch, seq_len, num_heads, head_dim]
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        
        # Reshape to [batch, seq_len, hidden_size]
        attn_output = attn_output.reshape([batch_size, seq_len, hidden_size])
        
        # Final output projection
        output = self.out_proj(attn_output)
        
        return output


class SigLIPMLP(Layer):
    """MLP block for SigLIP vision encoder."""
    
    def __init__(
        self,
        config: SigLIPVisionConfig,
        weights: Weights,
        layer_idx: int,
        dtype: DType,
        device: DeviceRef,
    ):
        self.config = config
        
        print(f"Initializing SigLIP MLP Layer {layer_idx}")
        
        # Try different MLP weight key patterns
        base_keys_to_try = [
            f"vision_tower.vision_model.encoder.layers.{layer_idx}.mlp",
            f"vision_model.encoder.layers.{layer_idx}.mlp", 
            f"vision_tower.encoder.layers.{layer_idx}.mlp",
            f"model.vision_tower.vision_model.encoder.layers.{layer_idx}.mlp",
        ]
        
        self.fc1 = None
        self.fc2 = None
        
        # Try to load actual MLP weights
        for base_key in base_keys_to_try:
            try:
                self.fc1 = Linear(
                    weights[f"{base_key}.fc1.weight"].allocate(
                        dtype, [config.intermediate_size, config.hidden_size], device=device
                    ),
                    weights[f"{base_key}.fc1.bias"].allocate(
                        dtype, [config.intermediate_size], device=device
                    ) if f"{base_key}.fc1.bias" in weights else None,
                )
                
                self.fc2 = Linear(
                    weights[f"{base_key}.fc2.weight"].allocate(
                        dtype, [config.hidden_size, config.intermediate_size], device=device
                    ),
                    weights[f"{base_key}.fc2.bias"].allocate(
                        dtype, [config.hidden_size], device=device  
                    ) if f"{base_key}.fc2.bias" in weights else None,
                )
                
                print(f"Loaded MLP weights from: {base_key}")
                break
                
            except (KeyError, AttributeError):
                continue
        
        # Create initialized MLPs if not found
        if self.fc1 is None:
            print(f"Creating initialized MLP for layer {layer_idx}")
            self._create_initialized_mlp(config, dtype, device)
        
        # Set activation function
        if config.hidden_act == "gelu":
            self.activation = ops.gelu
        elif config.hidden_act == "relu":
            self.activation = ops.relu
        elif config.hidden_act == "silu" or config.hidden_act == "swish":
            self.activation = ops.silu
        else:
            print(f"Unknown activation {config.hidden_act}, using GELU")
            self.activation = ops.gelu
    
    def _create_initialized_mlp(self, config, dtype, device):
        """Create properly initialized MLP layers."""
        import numpy as np
        from max.driver import Tensor
        
        # Xavier uniform for fc1
        bound1 = math.sqrt(6.0 / (config.hidden_size + config.intermediate_size))
        fc1_weight = np.random.uniform(
            -bound1, bound1, (config.intermediate_size, config.hidden_size)
        ).astype(np.float32)
        
        # Xavier uniform for fc2  
        bound2 = math.sqrt(6.0 / (config.intermediate_size + config.hidden_size))
        fc2_weight = np.random.uniform(
            -bound2, bound2, (config.hidden_size, config.intermediate_size)
        ).astype(np.float32)
        
        self.fc1 = Linear(Tensor.from_numpy(fc1_weight).to(device), None)
        self.fc2 = Linear(Tensor.from_numpy(fc2_weight).to(device), None)
    
    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Apply MLP transformation: hidden -> intermediate -> hidden."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLIPEncoderLayer(Layer):
    """Single transformer layer for SigLIP vision encoder."""
    
    def __init__(
        self,
        config: SigLIPVisionConfig,
        weights: Weights,
        layer_idx: int,
        dtype: DType,
        device: DeviceRef,
    ):
        self.layer_idx = layer_idx
        print(f"Initializing SigLIP Encoder Layer {layer_idx}")
        
        # Initialize attention and MLP components
        self.self_attn = SigLIPMultiHeadAttention(config, weights, layer_idx, dtype, device)
        self.mlp = SigLIPMLP(config, weights, layer_idx, dtype, device)
        print("self_attn :",self.self_attn)
        print("mlp :",self.mlp)
        # Try different layer norm key patterns
        base_keys_to_try = [
            f"vision_tower.vision_model.encoder.layers.{layer_idx}",
            f"vision_model.encoder.layers.{layer_idx}",
            f"vision_tower.encoder.layers.{layer_idx}",
            f"model.vision_tower.vision_model.encoder.layers.{layer_idx}",
        ]
        
        self.layer_norm1 = None
        self.layer_norm2 = None
        
        # Try to load actual layer norm weights
        for base_key in base_keys_to_try:
            try:
                print("base_key :",base_key)
                print("weights :",weights, )
                self.layer_norm1 = LayerNorm(
                    weights[f"{base_key}.layer_norm1.weight"].allocate(
                        dtype, [config.hidden_size], device=device
                    ),
                    weights[f"{base_key}.layer_norm1.bias"].allocate(
                        dtype, [config.hidden_size], device=device
                    ) if f"{base_key}.layer_norm1.bias" in weights else None,
                    eps=config.layer_norm_eps,
                )
                
                self.layer_norm2 = LayerNorm(
                    weights[f"{base_key}.layer_norm2.weight"].allocate(
                        dtype, [config.hidden_size], device=device
                    ),
                    weights[f"{base_key}.layer_norm2.bias"].allocate(
                        dtype, [config.hidden_size], device=device
                    ) if f"{base_key}.layer_norm2.bias" in weights else None,
                    eps=config.layer_norm_eps,
                )
                
                print(f"Loaded layer norms from: {base_key}")
                break
                
            except (KeyError, AttributeError):
                continue
        
        # Create initialized layer norms if not found
        if self.layer_norm1 is None:
            print(f"Creating initialized layer norms for layer {layer_idx}")
            self._create_initialized_layer_norms(config, dtype, device)
    
    def _create_initialized_layer_norms(self, config, dtype, device):
        """Create properly initialized layer normalizations."""
        import numpy as np
        from max.driver import Tensor
        
        # Layer norm weight initialized to 1, bias to 0
        weight_data = np.ones(config.hidden_size, dtype=np.float32)
        bias_data = np.zeros(config.hidden_size, dtype=np.float32)
        
        weight = Tensor.from_numpy(weight_data).to(device)
        bias = Tensor.from_numpy(bias_data).to(device)
        
        self.layer_norm1 = LayerNorm(weight, bias, eps=config.layer_norm_eps)
        
        # Second layer norm with same initialization
        weight2 = Tensor.from_numpy(weight_data).to(device)
        bias2 = Tensor.from_numpy(bias_data).to(device)
        
        self.layer_norm2 = LayerNorm(weight2, bias2, eps=config.layer_norm_eps)
    
    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Apply transformer layer with pre-norm and residual connections."""
        # Pre-norm attention block
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual
        
        # Pre-norm MLP block  
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states


class SigLIPVisionEncoder(Layer):
    """Complete SigLIP vision encoder."""
    
    def __init__(
        self,
        config: SigLIPVisionConfig,
        weights: Weights,
        dtype: DType,
        device: DeviceRef,
    ):
        self.config = config
        print("Initializing SigLIP Vision Encoder")
        print("SIGLIP_ENCODER: Creating patch embedding...")
        # Initialize patch embeddings
        self.embeddings = SigLIPPatchEmbedding(config, weights, dtype, device)
        print("SIGLIP_ENCODER: Patch embedding created successfully")
        # Try to load position embeddings
        pos_embed_keys_to_try = [
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "vision_model.embeddings.position_embedding", 
            "vision_tower.embeddings.position_embedding",
            "model.vision_tower.vision_model.embeddings.position_embedding",
        ]
        
        self.position_embedding = None
        # print("Available keys in weights:")
        # for key in weights.keys:
        #     print(key)
        print("Listing all weights keys:")
        for key, value in weights.items():
            print(key)

        for pos_key in pos_embed_keys_to_try:
            try:
                print("SIGLIP_ENCODER: Creating position embeddings...")
                self.position_embedding = weights[pos_key].allocate(
                    dtype, [config.num_patches, config.hidden_size], device=device
                )
                print(f"Loaded position embeddings from: {pos_key}")
                break
            except (KeyError, AttributeError):
                print(f"Position embedding key {pos_key} not found, trying next...")
                traceback.format_exc()
                continue
        
        # Create initialized position embeddings if not found
        if self.position_embedding is None:
            print("Creating initialized position embeddings")
            import numpy as np
            from max.driver import Tensor
            
            # Sinusoidal position embeddings
            pe = np.zeros((config.num_patches, config.hidden_size), dtype=np.float32)
            for pos in range(config.num_patches):
                for i in range(0, config.hidden_size, 2):
                    pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / config.hidden_size)))
                    if i + 1 < config.hidden_size:
                        pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / config.hidden_size)))
            
            self.position_embedding = Tensor.from_numpy(pe).to(device)
        
        # Create transformer layers - limit to reasonable number for stability
        self.layers = []
        num_layers_to_create = min(config.num_hidden_layers, 12)
        print(f"Creating {num_layers_to_create} transformer layers")
        
        for i in range(num_layers_to_create):
            
            start_time = time.time()
            print(f"Starting layer {i} creation...")
            print("config :", config,"weights :", weights, i,"dtype :", dtype, device)
            layer = SigLIPEncoderLayer(config, weights, i, dtype, device)
            self.layers.append(layer)
        
        # Try to load post layer norm
        post_ln_keys_to_try = [
            "vision_tower.vision_model.post_layernorm",
            "vision_model.post_layernorm",
            "vision_tower.post_layernorm", 
            "model.vision_tower.vision_model.post_layernorm",
        ]
        
        self.post_layernorm = None
        
        for post_ln_key in post_ln_keys_to_try:
            try:
                self.post_layernorm = LayerNorm(
                    weights[f"{post_ln_key}.weight"].allocate(
                        dtype, [config.hidden_size], device=device
                    ),
                    weights[f"{post_ln_key}.bias"].allocate(
                        dtype, [config.hidden_size], device=device
                    ) if f"{post_ln_key}.bias" in weights else None,
                    eps=config.layer_norm_eps,
                )
                print(f"Loaded post layer norm from: {post_ln_key}")
                break
            except (KeyError, AttributeError):
                continue
        
        # Create initialized post layer norm if not found
        if self.post_layernorm is None:
            print("Creating initialized post layer norm")
            import numpy as np
            from max.driver import Tensor
            
            weight_data = np.ones(config.hidden_size, dtype=np.float32)
            bias_data = np.zeros(config.hidden_size, dtype=np.float32)
            
            weight = Tensor.from_numpy(weight_data).to(device)
            bias = Tensor.from_numpy(bias_data).to(device)
            
            self.post_layernorm = LayerNorm(weight, bias, eps=config.layer_norm_eps)
        
        # Dropout
        if config.hidden_dropout > 0.0:
            self.dropout = lambda x: ops.dropout(x, config.hidden_dropout)
        else:
            self.dropout = None
        
        print("SigLIP Vision Encoder initialization complete")
    
    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        logger.debug("SigLIP Vision Encoder: Processing image...")
        print("In cross attention.py")
        print("SigLIP Vision Encoder: Processing image...")

        # Convert to patch embeddings
        embeddings = self.embeddings(pixel_values)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embedding
        
        # Apply embedding dropout if configured
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        
        # Apply transformer layers
        hidden_states = embeddings
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
        
        # Apply final layer norm
        hidden_states = self.post_layernorm(hidden_states)
        
        logger.debug(f"Vision encoding complete: {hidden_states.shape}")
        return hidden_states


class CrossModalProjector(Layer):
    """Projects vision features to language model embedding space."""
    
    def __init__(
        self,
        config: CrossModalProjectorConfig,
        weights: Weights,
        dtype: DType,
        device: DeviceRef,
    ):
        self.config = config
        print("Initializing Cross-Modal Projector")
        
        # Try different projector weight key patterns
        projector_keys_to_try = [
            "multi_modal_projector",
            "multimodal_projector",
            "vision_projector",
            "mm_projector",
            "model.multi_modal_projector",
        ]
        
        self.projector = None
        
        if config.projector_type == "linear":
            # Try to load linear projector weights
            for base_key in projector_keys_to_try:
                try:
                    weight_key = f"{base_key}.linear.weight" if ".linear." not in base_key else f"{base_key}.weight"
                    bias_key = f"{base_key}.linear.bias" if ".linear." not in base_key else f"{base_key}.bias"
                    
                    self.projector = Linear(
                        weights[weight_key].allocate(
                            dtype, 
                            [config.language_hidden_size, config.vision_hidden_size], 
                            device=device
                        ),
                        weights[bias_key].allocate(
                            dtype, [config.language_hidden_size], device=device
                        ) if bias_key in weights else None,
                    )
                    print(f"Loaded linear projector from: {base_key}")
                    break
                    
                except (KeyError, AttributeError):
                    continue
            
            # Create initialized projector if not found
            if self.projector is None:
                print("Creating initialized linear projector")
                import numpy as np
                from max.driver import Tensor
                
                # Xavier uniform initialization
                bound = math.sqrt(6.0 / (config.vision_hidden_size + config.language_hidden_size))
                weight_data = np.random.uniform(
                    -bound, bound, 
                    (config.language_hidden_size, config.vision_hidden_size)
                ).astype(np.float32)
                
                weight = Tensor.from_numpy(weight_data).to(device)
                self.projector = Linear(weight, None)
        
        elif config.projector_type == "mlp":
            # Multi-layer projector
            intermediate_size = config.intermediate_size or (config.vision_hidden_size * 2)
            
            # Try to load MLP projector weights
            for base_key in projector_keys_to_try:
                try:
                    self.fc1 = Linear(
                        weights[f"{base_key}.mlp.fc1.weight"].allocate(
                            dtype, [intermediate_size, config.vision_hidden_size], device=device
                        ),
                        weights[f"{base_key}.mlp.fc1.bias"].allocate(
                            dtype, [intermediate_size], device=device
                        ) if f"{base_key}.mlp.fc1.bias" in weights else None,
                    )
                    
                    self.fc2 = Linear(
                        weights[f"{base_key}.mlp.fc2.weight"].allocate(
                            dtype, [config.language_hidden_size, intermediate_size], device=device
                        ),
                        weights[f"{base_key}.mlp.fc2.bias"].allocate(
                            dtype, [config.language_hidden_size], device=device
                        ) if f"{base_key}.mlp.fc2.bias" in weights else None,
                    )
                    
                    print(f"Loaded MLP projector from: {base_key}")
                    break
                    
                except (KeyError, AttributeError):
                    continue
            
            # Create initialized MLP projector if not found
            if not hasattr(self, 'fc1'):
                print("Creating initialized MLP projector")
                self._create_initialized_mlp_projector(config, dtype, device, intermediate_size)
            
            # Set activation function
            if config.hidden_act == "gelu":
                self.activation = ops.gelu
            elif config.hidden_act == "relu": 
                self.activation = ops.relu
            elif config.hidden_act == "silu":
                self.activation = ops.silu
            else:
                print(f"Unknown activation {config.hidden_act}, using GELU")
                self.activation = ops.gelu
        
        else:
            raise ValueError(f"Unsupported projector type: {config.projector_type}")
        
        print(f"Cross-Modal Projector ({config.projector_type}) initialized")
    
    def _create_initialized_mlp_projector(self, config, dtype, device, intermediate_size):
        """Create properly initialized MLP projector.""" 
        import numpy as np
        from max.driver import Tensor
        
        # Xavier uniform for fc1
        bound1 = math.sqrt(6.0 / (config.vision_hidden_size + intermediate_size))
        fc1_weight = np.random.uniform(
            -bound1, bound1, (intermediate_size, config.vision_hidden_size)
        ).astype(np.float32)
        
        # Xavier uniform for fc2
        bound2 = math.sqrt(6.0 / (intermediate_size + config.language_hidden_size))
        fc2_weight = np.random.uniform(
            -bound2, bound2, (config.language_hidden_size, intermediate_size)
        ).astype(np.float32)
        
        self.fc1 = Linear(Tensor.from_numpy(fc1_weight).to(device), None)
        self.fc2 = Linear(Tensor.from_numpy(fc2_weight).to(device), None)
    
    def __call__(self, vision_features: TensorValue) -> TensorValue:
        """Project vision features to language embedding space."""
        logger.debug(f"Projecting vision features: {vision_features.shape}")
        
        if self.config.projector_type == "linear":
            projected = self.projector(vision_features)
            
        elif self.config.projector_type == "mlp":
            hidden = self.fc1(vision_features)
            hidden = self.activation(hidden)
            projected = self.fc2(hidden)
        
        logger.debug(f"Projection complete: {vision_features.shape} -> {projected.shape}")
        return projected
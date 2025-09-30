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
from max.graph.type import ConvInputLayout, FilterLayout
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
        
        # ✅ CORRECT: Use exact weight keys from your model
        weight_key = "vision_tower.vision_model.embeddings.patch_embedding.weight"
        bias_key = "vision_tower.vision_model.embeddings.patch_embedding.bias"
        
        self.weight = None
        self.bias = None
        
        # Try to load actual weights with exact keys
        try:
            self.weight = weights[weight_key].allocate(
                dtype,
                [config.hidden_size, config.num_channels, config.patch_size, config.patch_size],
                device=device,
            )
            print(f"✅ Loaded patch embedding weights from: {weight_key}")
        except (KeyError, AttributeError) as e:
            print(f"⚠️ Patch embedding weight not found: {e}")
        
        # Try to load bias
        try:
            self.bias = weights[bias_key].allocate(
                dtype,
                [config.hidden_size],
                device=device,
            )
            print(f"✅ Loaded patch embedding bias from: {bias_key}")
        except (KeyError, AttributeError) as e:
            print(f"⚠️ Patch embedding bias not found: {e}")
        
        # If no actual weights found, create learnable parameters with proper initialization
        if self.weight is None:
            print("Creating initialized patch embedding weights")
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
            import numpy as np
            from max.driver import Tensor
            bias_data = np.zeros(config.hidden_size, dtype=np.float32)
            self.bias = Tensor.from_numpy(bias_data).to(device)
    
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
            padding=[0, 0, 0, 0],       # Four values: [pad_h_before, pad_w_before, pad_h_after, pad_w_after]
            input_layout=ConvInputLayout.NCHW,         # Match your tensor format
            filter_layout=FilterLayout.FCRS,        # [out, in, h, w]
        )
        
        # Get dimensions
        print(f"Original Patch embeddings shape: {embeddings.shape}")

        batch_size = embeddings.shape[0]
        height_patches = embeddings.shape[1]
        width_patches = embeddings.shape[2]
        hidden_size = embeddings.shape[3]
        # Reshape to [batch, hidden_size, num_patches]
        num_patches = height_patches * width_patches
        print("num_patches :",num_patches)
        embeddings = embeddings.reshape((batch_size, num_patches, hidden_size))
        print(f"Patch embeddings shape: {embeddings.shape}")
        
        # Transpose to [batch, num_patches, hidden_size] (standard transformer format)
        # embeddings = ops.transpose(embeddings, 1, 2)
        
        return embeddings


class SigLIPMultiHeadAttention(Layer):    
    def __init__(self, config, weights, layer_idx, dtype, device):
        
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
        
        # ✅ CORRECT: Use exact weight keys from your model
        base_key = f"vision_tower.vision_model.encoder.layers.{layer_idx}.self_attn"
        
        print(f"[LAYER {layer_idx}] Loading attention weights from: {base_key}")
        
        try:
            self._load_weights(base_key, weights, config, dtype, device)
            print(f"[LAYER {layer_idx}] ✅ SUCCESS with {base_key}")
        except Exception as e:
            print(f"[LAYER {layer_idx}] ❌ FAILED {base_key}: {e}")
            traceback.print_exc()
            print(f"[LAYER {layer_idx}] Creating initialized attention projections")
            self._create_initialized_projections(config, dtype, device)

    def _load_weights(self, base_key, weights, config, dtype, device):
        # ✅ Load weights with proper error handling
        def safe_alloc_weight(name, shape):
            start = time.time()
            try:
                # ✅ CORRECT: Map out_proj to exact key from your weights
                actual_name = "out_proj" if name == "out_proj" else name
                w = weights[f"{base_key}.{actual_name}.weight"].allocate(dtype, shape, device=device)
                elapsed = time.time() - start
                print(f"[LAYER {self.layer_idx}] {name} -> {actual_name} weight allocated in {elapsed:.2f}s")
                return w
            except Exception as e:
                elapsed = time.time() - start
                print(f"[LAYER {self.layer_idx}] {name} weight FAILED in {elapsed:.2f}s: {e}")
                raise
        
        def safe_alloc_bias(name, shape):
            actual_name = "out_proj" if name == "out_proj" else name
            bias_key = f"{base_key}.{actual_name}.bias"
            start = time.time()
            try:
                b = weights[bias_key].allocate(dtype, shape, device=device)
                elapsed = time.time() - start
                print(f"[LAYER {self.layer_idx}] {name} -> {actual_name} bias allocated in {elapsed:.2f}s")
                return b
            except Exception as e:
                elapsed = time.time() - start
                print(f"[LAYER {self.layer_idx}] {name} bias FAILED in {elapsed:.2f}s: {e}")
                return None
        
        print("dtype :", dtype, "device :", device)
        
        # ✅ Create Linear layers with proper MAX framework usage
        self.q_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.hidden_size, 
            dtype=dtype,
            device=device,
            has_bias=True
        )
        self.q_proj.weight = safe_alloc_weight("q_proj", [config.hidden_size, config.hidden_size])
        self.q_proj.bias = safe_alloc_bias("q_proj", [config.hidden_size])
        
        self.k_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True
        )
        self.k_proj.weight = safe_alloc_weight("k_proj", [config.hidden_size, config.hidden_size])
        self.k_proj.bias = safe_alloc_bias("k_proj", [config.hidden_size])

        self.v_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True
        )
        self.v_proj.weight = safe_alloc_weight("v_proj", [config.hidden_size, config.hidden_size])
        self.v_proj.bias = safe_alloc_bias("v_proj", [config.hidden_size])

        # ✅ CRITICAL: Use "out_proj" which gets mapped to "out_proj"
        self.out_proj = Linear(
            in_dim=config.hidden_size,
            out_dim=config.hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True
        )
        self.out_proj.weight = safe_alloc_weight("out_proj", [config.hidden_size, config.hidden_size])
        self.out_proj.bias = safe_alloc_bias("out_proj", [config.hidden_size])
    
    def _create_initialized_projections(self, config, dtype, device):
        """Create properly initialized attention projections."""
        import numpy as np
        from max.driver import Tensor
        
        # Xavier uniform initialization
        bound = math.sqrt(6.0 / (2 * config.hidden_size))
        
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            projection = Linear(
                in_dim=config.hidden_size,
                out_dim=config.hidden_size,
                dtype=dtype,
                device=device,
                has_bias=True
            )
            
            weight_data = np.random.uniform(
                -bound, bound, (config.hidden_size, config.hidden_size)
            ).astype(np.float32)
            
            # Replace the layer's weight with initialized tensor
            projection.weight = Tensor.from_numpy(weight_data).to(device)
            
            # Set the projection as attribute
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
        query = ops.transpose(query, 1, 2)
        key   = ops.transpose(key,   1, 2)
        value = ops.transpose(value, 1, 2)
        
        # Scaled dot-product attention
        scores = ops.matmul(query, ops.transpose(key,   2, 3)) * self.scale
        
        # Apply softmax
        attn_weights = ops.softmax(scores)
        
        # Apply dropout if specified
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = ops.matmul(attn_weights, value)
        
        # Transpose back to [batch, seq_len, num_heads, head_dim]
        attn_output = ops.transpose(attn_output, 1, 2)
        
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
        
        # ✅ CORRECT: Use exact weight keys from your model
        base_key = f"vision_tower.vision_model.encoder.layers.{layer_idx}.mlp"
        
        try:
            # ✅ Create Linear layers with proper MAX framework usage
            self.fc1 = Linear(
                in_dim=config.hidden_size,
                out_dim=config.intermediate_size,
                dtype=dtype,
                device=device,
                has_bias=True
            )
            self.fc1.weight = weights[f"{base_key}.fc1.weight"].allocate(
                dtype, [config.intermediate_size, config.hidden_size], device=device
            )
            
            # Try to load bias
            self.fc1.bias = weights[f"{base_key}.fc1.bias"].allocate(
                dtype, [config.intermediate_size], device=device
            )
            
            self.fc2 = Linear(
                in_dim=config.intermediate_size,
                out_dim=config.hidden_size,
                dtype=dtype,
                device=device,
                has_bias=True
            )
            self.fc2.weight = weights[f"{base_key}.fc2.weight"].allocate(
                dtype, [config.hidden_size, config.intermediate_size], device=device
            )
            
            # Try to load bias
            self.fc2.bias = weights[f"{base_key}.fc2.bias"].allocate(
                dtype, [config.hidden_size], device=device
            )
            
            print(f"✅ Loaded MLP weights from: {base_key}")
                
        except (KeyError, AttributeError) as e:
            print(f"⚠️ MLP weights not found for layer {layer_idx}: {e}")
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
        
        self.fc1 = Linear(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=True
        )
        self.fc1.weight = Tensor.from_numpy(fc1_weight).to(device)
        
        self.fc2 = Linear(
            in_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            dtype=dtype,
            device=device,
            has_bias=True
        )
        self.fc2.weight = Tensor.from_numpy(fc2_weight).to(device)
    
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
        
        # ✅ CORRECT: Use exact weight keys from your model
        base_key = f"vision_tower.vision_model.encoder.layers.{layer_idx}"
        
        # Try to load actual layer norm weights with exact keys
        try:
            ln1_weight = weights[f"{base_key}.layer_norm1.weight"].allocate(
                dtype, [config.hidden_size], device=device
            )
            ln1_bias = None
            try:
                ln1_bias = weights[f"{base_key}.layer_norm1.bias"].allocate(
                    dtype, [config.hidden_size], device=device
                )
            except (KeyError, AttributeError):
                pass
            
            # ✅ FIXED: Create LayerNorm with proper constructor, then assign weights
            self.layer_norm1 = LayerNorm(
                config.hidden_size,  # dims
                device,              # device
                dtype,               # dtype
                config.layer_norm_eps,  # eps
                True                 # use_bias
            )
            self.layer_norm1.weight = ln1_weight
            if ln1_bias is not None:
                self.layer_norm1.bias = ln1_bias
            
            ln2_weight = weights[f"{base_key}.layer_norm2.weight"].allocate(
                dtype, [config.hidden_size], device=device
            )
            ln2_bias = None
            try:
                ln2_bias = weights[f"{base_key}.layer_norm2.bias"].allocate(
                    dtype, [config.hidden_size], device=device
                )
            except (KeyError, AttributeError):
                pass
            
            # ✅ FIXED: Create LayerNorm with proper constructor, then assign weights
            self.layer_norm2 = LayerNorm(
                config.hidden_size,  # dims
                device,              # device
                dtype,               # dtype
                config.layer_norm_eps,  # eps
                True                 # use_bias
            )
            self.layer_norm2.weight = ln2_weight
            if ln2_bias is not None:
                self.layer_norm2.bias = ln2_bias
            
            print(f"✅ Loaded layer norms from: {base_key}")
                
        except (KeyError, AttributeError) as e:
            print(f"⚠️ Layer norms not found for layer {layer_idx}: {e}")
            print(f"Creating initialized layer norms for layer {layer_idx}")
            self._create_initialized_layer_norms(config, dtype, device)
    
    def _create_initialized_layer_norms(self, config, dtype, device):
        """Create properly initialized layer normalizations."""
        import numpy as np
        from max.driver import Tensor
        
        # ✅ FIXED: Create LayerNorm with proper constructor, then assign initialized weights
        self.layer_norm1 = LayerNorm(
            config.hidden_size,  # dims
            device,              # device
            dtype,               # dtype
            config.layer_norm_eps,  # eps
            True                 # use_bias
        )
        
        # Layer norm weight initialized to 1, bias to 0
        weight_data = np.ones(config.hidden_size, dtype=np.float32)
        bias_data = np.zeros(config.hidden_size, dtype=np.float32)
        
        self.layer_norm1.weight = Tensor.from_numpy(weight_data).to(device)
        self.layer_norm1.bias = Tensor.from_numpy(bias_data).to(device)
        
        # Second layer norm with same initialization
        self.layer_norm2 = LayerNorm(
            config.hidden_size,  # dims
            device,              # device
            dtype,               # dtype
            config.layer_norm_eps,  # eps
            True                 # use_bias
        )
        
        self.layer_norm2.weight = Tensor.from_numpy(weight_data).to(device)
        self.layer_norm2.bias = Tensor.from_numpy(bias_data).to(device)
    
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
        
        # Initialize patch embeddings
        print("SIGLIP_ENCODER: Creating patch embedding...")
        self.embeddings = SigLIPPatchEmbedding(config, weights, dtype, device)
        print(self.embeddings)
        print("SIGLIP_ENCODER: Patch embedding created successfully")
        
        # ✅ CORRECT: Use exact weight key from your model
        pos_embed_key = "vision_tower.vision_model.embeddings.position_embedding.weight"
        
        print("SIGLIP_ENCODER: Creating position embeddings...")
        try:
            self.position_embedding = weights[pos_embed_key].allocate(
                dtype, [config.num_patches, config.hidden_size], device=device
            )
            print(f"✅ Loaded position embeddings from: {pos_embed_key}")
        except (KeyError, AttributeError) as e:
            print(f"⚠️ Position embeddings not found: {e}")
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
        num_layers_to_create = min(config.num_hidden_layers, 27)  # All layers based on your weights
        print(f"Creating {num_layers_to_create} transformer layers")
        
        for i in range(num_layers_to_create):
            layer = SigLIPEncoderLayer(config, weights, i, dtype, device)
            self.layers.append(layer)
        
        # ✅ CORRECT: Use exact weight key from your model
        post_ln_key = "vision_tower.vision_model.post_layernorm"
        
        try:
            ln_weight = weights[f"{post_ln_key}.weight"].allocate(
                dtype, [config.hidden_size], device=device
            )
            ln_bias = None
            try:
                ln_bias = weights[f"{post_ln_key}.bias"].allocate(
                    dtype, [config.hidden_size], device=device
                )
            except (KeyError, AttributeError):
                pass
            
            # ✅ FIXED: Create LayerNorm with proper constructor, then assign weights
            self.post_layernorm = LayerNorm(
                config.hidden_size,  # dims
                device,              # device
                dtype,               # dtype
                config.layer_norm_eps,  # eps
                True                 # use_bias
            )
            self.post_layernorm.weight = ln_weight
            if ln_bias is not None:
                self.post_layernorm.bias = ln_bias
            
            print(f"✅ Loaded post layer norm from: {post_ln_key}")
        except (KeyError, AttributeError) as e:
            print(f"⚠️ Post layer norm not found: {e}")
            print("Creating initialized post layer norm")
            import numpy as np
            from max.driver import Tensor
            
            # ✅ FIXED: Create LayerNorm with proper constructor, then assign initialized weights
            self.post_layernorm = LayerNorm(
                config.hidden_size,  # dims
                device,              # device
                dtype,               # dtype
                config.layer_norm_eps,  # eps
                True                 # use_bias
            )
            
            weight_data = np.ones(config.hidden_size, dtype=np.float32)
            bias_data = np.zeros(config.hidden_size, dtype=np.float32)
            
            self.post_layernorm.weight = Tensor.from_numpy(weight_data).to(device)
            self.post_layernorm.bias = Tensor.from_numpy(bias_data).to(device)
        
        # Dropout
        if config.hidden_dropout > 0.0:
            self.dropout = lambda x: ops.dropout(x, config.hidden_dropout)
        else:
            self.dropout = None
        
        print("SigLIP Vision Encoder initialization complete")
    
    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        print("SigLIP Vision Encoder: Processing image...")
        embeddings = self.embeddings(pixel_values)
        # print(f"Patch embeddings shape: {embeddings.shape}")   
        # Ensure position embedding is compatible: (num_patches, hidden_size)
        position_embedding = self.position_embedding
        # if position_embedding.shape != (h * w, c):
        #     raise ValueError(f"Position embedding shape mismatch: {position_embedding.shape} vs {embeddings.shape}")
        print(f"Position embedding shape: {position_embedding.shape}")
        position_embedding = ops.reshape(position_embedding, (1, 4096, 1152))
  # Add batch dim
        
        # dtype casting if needed
        if embeddings.dtype != position_embedding.dtype:
            position_embedding = ops.cast(position_embedding, embeddings.dtype)
        
        embeddings = embeddings + position_embedding
                
        # Apply dropout if configured
        if self.dropout is not None:
            embeddings = self.dropout(embeddings)
        
        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final layer norm
        hidden_states = self.post_layernorm(hidden_states)
        
        print(f"Vision encoding completed: {hidden_states.shape}")
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
        
        # ✅ CORRECT: Use exact weight keys from your model
        # From your weights: multimodalprojector.mminputprojectionweight, multimodalprojector.mmsoftembnorm.weight
        projector_weight_key = "multi_modal_projector.mm_input_projection_weight"
        soft_emb_norm_key = "multi_modal_projector.mm_soft_emb_norm.weight"
        
        self.projector = None
        self.soft_emb_norm = None
        
        if config.projector_type == "linear":
            # Try to load projector weights with exact keys
            try:
                projector_weight = weights[projector_weight_key].allocate(
                    dtype,
                    [config.vision_hidden_size, config.language_hidden_size],
                    device=device
                )
                self.projector = Linear(
                    in_dim=config.vision_hidden_size,
                    out_dim=config.language_hidden_size,
                    dtype=dtype,
                    device=device,
                    has_bias=False
                )
                self.projector.weight = projector_weight
                
                print(f"✅ Loaded linear projector from: {projector_weight_key}")
                
                # Try to load soft embedding norm
                try:
                    print("language_hidden_size :",config.language_hidden_size)
                    print("vision_hidden_size :",config.vision_hidden_size)
                    soft_emb_weight = weights[soft_emb_norm_key].allocate(
                        dtype, [config.vision_hidden_size], device=device
                    )
                    
                    # ✅ FIXED: Create LayerNorm with proper constructor, then assign weights
                    self.soft_emb_norm = LayerNorm(
                        config.vision_hidden_size,  # dims
                        device,                       # device
                        dtype,                        # dtype
                        1e-6,                        # eps
                        False                        # use_bias (no bias in your weights)
                    )
                    self.soft_emb_norm.weight = soft_emb_weight
                    
                    print(f"✅ Loaded soft embedding norm from: {soft_emb_norm_key}")
                except (KeyError, AttributeError) as e:
                    print(f"⚠️ Soft embedding norm not found: {e}")
                    
            except (KeyError, AttributeError) as e:
                print(f"⚠️ Projector weights not found: {e}")
                print("Creating initialized linear projector")
                self._create_initialized_projector(config, dtype, device)
        
        elif config.projector_type == "mlp":
            # Multi-layer projector - not found in your weights, so initialize
            print("Creating initialized MLP projector")
            intermediate_size = config.intermediate_size or (config.vision_hidden_size * 2)
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
    
    def _create_initialized_projector(self, config, dtype, device):
        """Create initialized linear projector."""
        import numpy as np
        from max.driver import Tensor
        
        # Xavier uniform initialization
        bound = math.sqrt(6.0 / (config.vision_hidden_size + config.language_hidden_size))
        weight_data = np.random.uniform(
            -bound, bound, 
            (config.language_hidden_size, config.vision_hidden_size)
        ).astype(np.float32)
        
        self.projector = Linear(
            in_dim=config.vision_hidden_size,
            out_dim=config.language_hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False
        )
        self.projector.weight = Tensor.from_numpy(weight_data).to(device)
    
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
        
        self.fc1 = Linear(
            in_dim=config.vision_hidden_size,
            out_dim=intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=False
        )
        self.fc1.weight = Tensor.from_numpy(fc1_weight).to(device)
        
        self.fc2 = Linear(
            in_dim=intermediate_size,
            out_dim=config.language_hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False
        )
        self.fc2.weight = Tensor.from_numpy(fc2_weight).to(device)
    
    def __call__(self, vision_features: TensorValue) -> TensorValue:
        """Project vision features to language embedding space."""
        logger.debug(f"Projecting vision features: {vision_features.shape}")
        
        if self.config.projector_type == "linear":
            projected = self.projector(vision_features)
            
            # Apply soft embedding norm if available
            if self.soft_emb_norm is not None:
                projected = self.soft_emb_norm(projected)
                
        elif self.config.projector_type == "mlp":
            hidden = self.fc1(vision_features)
            hidden = self.activation(hidden)
            projected = self.fc2(hidden)
        
        logger.debug(f"Projection complete: {vision_features.shape} -> {projected.shape}")
        return projected
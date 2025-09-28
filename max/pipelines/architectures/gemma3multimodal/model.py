from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Sequence

from max.driver import Device, CPU, Accelerator, accelerator_count, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    SupportedEncoding,
    ModelInputs,  # Added this import
)
from transformers import AutoConfig
from max.pipelines.core import TextAndVisionContext


# Import the base Gemma3Model to inherit from
from max.pipelines.architectures.gemma3.model import Gemma3Model
import torch
from PIL import Image

# Set up logging properly
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import your multimodal components
from .model_config import Gemma3MultimodalConfig, SigLIPVisionConfig, CrossModalProjectorConfig
from .cross_attention import SigLIPVisionEncoder, CrossModalProjector
from .graph_builder import build_multimodal_graphs


class Gemma3MultimodalModel(Gemma3Model):
    """Gemma3 Multimodal model with vision processing."""
    
    _strict_state_dict_loading = False
    
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        return_logits = None,
    ) -> None:
        self.session = session
        print("=== Gemma3MultimodalModel __init__ called ===")
        
        # Extract text config for parent
        text_config = getattr(huggingface_config, "text_config", huggingface_config)
        
        # Initialize parent Gemma3Model FIRST
        super().__init__(
            pipeline_config, session, text_config, encoding, devices,
            kv_cache_config, weights, adapter, return_logits,
        )
        
        print(f"Model dtype after super: {self.dtype}")
        
        # Skip weight debugging for now to avoid the error
        print("=== SKIPPING WEIGHT DEBUGGING ===")
        print("=" * 50)
        
        # Create proper multimodal config
        self.multimodal_config = self._build_multimodal_config(huggingface_config)
        
        # Don't initialize vision components here - they need to be built within graph context
        self.vision_encoder = None
        self.projector = None
        self._vision_graph_built = False
        self.vision_graph = None
        self.vision_model = None
        
        print("✅ Multimodal model initialized (vision components will be built on first use)")
        
    def _build_multimodal_config(self, hf_config: AutoConfig) -> Gemma3MultimodalConfig:
        """Build complete multimodal config from HuggingFace config."""
        
        # Build vision config
        vision_config = SigLIPVisionConfig()
        if hasattr(hf_config, 'vision_config'):
            for key, value in hf_config.vision_config.__dict__.items():
                if hasattr(vision_config, key):
                    setattr(vision_config, key, value)
        
        # Build projector config
        projector_config = CrossModalProjectorConfig(
            vision_hidden_size=1152,
            language_hidden_size=3072,
            projector_type="linear",
            num_layers=1,
        )
        
        # Create complete multimodal config
        multimodal_config = Gemma3MultimodalConfig(
            vision_config=vision_config,
            projector_config=projector_config,
            text_config=hf_config.text_config if hasattr(hf_config, 'text_config') else None
        )
        
        return multimodal_config

    # def _build_vision_components_if_needed(self):
    #     if not self._vision_graph_built:
    #         try:
    #             print("🔧 Building vision components within graph context...")
                
    #             # Use your existing graph builder
    #             from .graph_builder import VisionGraphBuilder
                
    #             builder = VisionGraphBuilder(
    #                 config=self.multimodal_config,  # Pass complete multimodal config
    #                 weights=self.weights,  # This should be accessible from parent
    #                 dtype=self.dtype,
    #                 device=self.devices[0],
    #             )
                
    #             # Build vision graph and STORE IT
    #             self.vision_graph = builder.build_vision_graph()  # ← This was missing!
                
    #             # Mark as built
    #             self._vision_graph_built = True
    #             print("✅ Vision graph built successfully")
                
    #         except Exception as e:
    #             print(f"❌ Failed to build vision components: {e}")
    #             import traceback
    #             traceback.print_exc()

    def _build_vision_components_if_needed(self):
        if self.vision_graph is None:  # Always try if None
            print("🔧 Starting vision graph building process...")
            
            try:
                self.vision_graph, self.language_graph = build_multimodal_graphs(
                config=self.multimodal_config,
                weights=self.weights,
                dtype=self.dtype,
                device=self.devices[0],
                optimize=True
                )
                self._vision_graph_built = True
                print("🎉 Vision graph building completed successfully!")
                
            except Exception as e:
                print(f"❌ Graph creation failed: {e}")
                print(f"❌ Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                
                # Don't set the graph on failure
                self.vision_graph = None
        else:
            print("📋 Vision graph already exists, skipping...")

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs=None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        print("=" * 80)
        print("=== MULTIMODAL PREPARE_INITIAL_TOKEN_INPUTS CALLED ===")
        print("=" * 80)
        
        # Check if we have vision inputs
        vision_tokens_np = None
        has_vision = any(hasattr(ctx, 'pixel_values') and ctx.pixel_values 
                        for ctx in context_batch)
        
        if has_vision:
            print("🖼️ Vision input detected, building vision components...")
            self._build_vision_components_if_needed()
            
            # Check if vision graph was built successfully
            if hasattr(self, 'vision_graph') and self.vision_graph is not None:
                print("✅ Vision graph exists, proceeding with compilation...")
                
                # Load (compile) the vision graph
                if not hasattr(self, "vision_model") or self.vision_model is None:
                    print("🎯 Loading/compiling vision graph...")
                    
                    # Use same device as graph
                    device = CPU() if accelerator_count() == 0 else Accelerator()
                    vision_session = InferenceSession(devices=[device])
                    self.vision_model = vision_session.load(self.vision_graph)
                    print("✅ Vision graph loaded and compiled")
                else:
                    print("📋 Vision model already compiled, skipping...")
                print("🚀 Executing vision model on image data...")
                for i, ctx in enumerate(context_batch):
                    if hasattr(ctx, 'pixel_values') and ctx.pixel_values:
                        try:
                            import numpy as np
                            from max.driver import Tensor
                            from PIL import Image
                            # Get and preprocess the image
                            pixel_array = ctx.pixel_values[0].astype(np.float32)
                            
                            # Resize to expected input size (896x896)
                            if pixel_array.shape[:2] != (896, 896):
                                pil_img = Image.fromarray(pixel_array.astype(np.uint8))
                                pil_img = pil_img.resize((896, 896))
                                pixel_array = np.array(pil_img).astype(np.float32)
                            
                            # Normalize and reshape: [H,W,C] -> [1,C,H,W]
                            pixel_array = (pixel_array / 127.5) - 1.0
                            pixel_array = pixel_array.transpose(2, 0, 1)[None]  # [1,C,H,W]
                            
                            print(f"📸 Preprocessed image tensor shape: {pixel_array.shape}")
                            
                            # Create MAX Tensor and execute
                            device = CPU() if accelerator_count() == 0 else Accelerator()
                            pixel_tensor = Tensor.from_numpy(pixel_array).to(device)
                            
                            # Execute the vision model
                            vision_outputs = self.vision_model.execute(pixel_tensor)
                            vision_tokens = vision_outputs[0]
                            
                            if hasattr(vision_tokens, 'to'):
                                vision_tokens = vision_tokens.to(CPU())
                            
                            print(f"✅ Got vision tokens shape: {vision_tokens.shape}")
                            print(f"🎯 Vision tokens tensor type: {type(vision_tokens)}")
                            
                            # TODO: Integrate vision_tokens into text sequence
                            # This is where you'd prepend/insert the vision tokens into the text input
                            vision_tokens_result = vision_tokens
                            break  # Process only first image for now
                            
                        except Exception as e:
                            print(f"❌ Error executing vision model on image {i}: {e}")
                            import traceback
                            traceback.print_exc()
                            vision_tokens_result = None
            else:           
                print("❌ No vision graph available, skipping vision processing")
                
        else:
            print("📝 Text-only input, skipping vision processing")
        
        # Get text inputs from parent
        if 'vision_tokens_result' in locals() and vision_tokens_result is not None:
            # Convert MAX tensor to numpy for integration
            vision_tokens_np = vision_tokens_result.to_numpy()  # ← Use the stored result
            print(f"🔗 Converting vision tokens for integration: {vision_tokens_np.shape}")
        else:
            vision_tokens_np = None
            print("❌ No vision tokens available for integration")
        print(f"🔗 Converting vision tokens for integration: {vision_tokens_np.shape}")
        text_inputs = super().prepare_initial_token_inputs(
            context_batch, kv_cache_inputs, return_n_logits
        )
        if vision_tokens_np is not None:
            print("🔀 Integrating vision and text tokens...")
            
            # Get the text token IDs
        #     if hasattr(text_inputs, 'input_ids'):
        #         token_ids = text_inputs.input_ids
        #         print(f"✅ Found input_ids: {token_ids.shape}")
        #     elif hasattr(text_inputs, 'tokens'):
        #         token_ids = text_inputs.tokens
        #         print(f"✅ Found tokens: {token_ids.shape}")
        #     elif hasattr(text_inputs, 'input_tokens'):
        #         token_ids = text_inputs.input_tokens
        #         print(f"✅ Found input_tokens: {token_ids.shape}")
        #     else:
        #         print("❌ Could not find token IDs in text_inputs")
        #         print(f"Available attributes: {[attr for attr in dir(text_inputs) if not attr.startswith('_')]}")
        #         print("All attributes:", text_inputs.__dict__)
                
        #         # For now, just log and return unmodified inputs
        #         print("⚠️ Returning unmodified text inputs for now")
        #         return text_inputs
            
        #     # Store vision embeddings for later use (if possible)
        #     try:
        #         text_inputs.vision_embeddings = vision_tokens_np
        #         text_inputs.num_vision_tokens = vision_tokens_np.shape[1]
        #         print(f"✅ Stored vision embeddings in text_inputs")
        #     except Exception as e:
        #         print(f"⚠️ Could not store vision embeddings: {e}")
            
        #     print(f"🎯 Vision integration attempted!")
        
        # print("=== MULTIMODAL PREPARE_INITIAL_TOKEN_INPUTS COMPLETED ===")
        # return text_inputs

            text_tokens = text_inputs.tokens  # Shape: (text_seq_len,)
            text_seq_len = text_tokens.shape[0]
            
            print(f"📊 Text tokens shape: {text_tokens.shape}")
            print(f"📊 Vision tokens shape: {vision_tokens_np.shape}")
            
            # Create vision token IDs (using special token ID for vision patches)
            vision_token_id = 32000  # Special token ID for vision
            num_vision_patches = vision_tokens_np.shape[1]  # 2048
            
            # Create vision token IDs array: [num_vision_patches]
            if hasattr(text_tokens, 'to_numpy'):
                text_tokens_np = text_tokens.to_numpy()
                numpy_dtype = text_tokens_np.dtype  # This will be numpy.int64 or similar
            else:
                text_tokens_np = text_tokens
                numpy_dtype = text_tokens_np.dtype
            import numpy as np
            vision_token_ids = np.full((num_vision_patches,), vision_token_id, dtype=numpy_dtype)
            print(f"🔢 Vision token IDs shape: {vision_token_ids.shape}, dtype: {vision_token_ids.dtype}")
            print(f"🔢 Text tokens shape: {text_tokens_np.shape}, dtype: {text_tokens_np.dtype}")
    
            # Concatenate vision tokens BEFORE text tokens: [vision_tokens, text_tokens]
                
            combined_tokens = np.concatenate([vision_token_ids, text_tokens_np])
            
            print(f"🎯 Combined token sequence: vision({num_vision_patches}) + text({text_seq_len}) = {combined_tokens.shape}")
            
            # Convert back to MAX Tensor if original was MAX Tensor
            if hasattr(text_tokens, 'to_numpy'):
                from max.driver import Tensor
                combined_tensor = Tensor.from_numpy(combined_tokens)
                if hasattr(text_tokens, 'device'):
                    combined_tensor = combined_tensor.to(text_tokens.device)
                text_inputs.tokens = combined_tensor
            else:
                text_inputs.tokens = combined_tokens
            
            # Store vision embeddings and metadata for the model to use
            text_inputs.vision_embeddings = vision_tokens_np.squeeze(0)  # Remove batch dim: (2048, 1176)
            text_inputs.num_vision_tokens = num_vision_patches
            text_inputs.vision_token_id = vision_token_id
            
            print(f"✅ Successfully integrated vision tokens!")
            print(f"🎯 Final token sequence shape: {text_inputs.tokens.shape}")
            print(f"🎯 Vision embeddings stored: {text_inputs.vision_embeddings.shape}")
            
        print("=== MULTIMODAL PREPARE_INITIAL_TOKEN_INPUTS COMPLETED ===")
        return text_inputs



    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs=None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        print("=" * 80)
        print("=== MULTIMODAL PREPARE_INITIAL_TOKEN_INPUTS CALLED ===")
        print("=" * 80)

        # Store vision features temporarily
        vision_features_storage = None
        num_vision_tokens_storage = 0

        # Check if we have vision inputs
        has_vision = any(hasattr(ctx, 'pixel_values') and ctx.pixel_values
                        for ctx in context_batch)

        if has_vision:
            print("🖼️ Vision input detected, building vision components...")
            self._build_vision_components_if_needed()

            # Use the model's session and weights for compilation
            if not hasattr(self, "vision_model") or self.vision_model is None:
                print("🎯 Loading/compiling vision graph...")
                try:
                    if not hasattr(self, "vision_graph") or self.vision_graph is None:
                        self.vision_graph = self._build_vision_graph()
                    self.vision_model = self.session.load(
                        self.vision_graph,
                        weights_registry=self.weights.allocated_weights,
                    )
                    print("✅ Vision graph loaded and compiled")
                except Exception as e:
                    print(f"❌ Failed to compile vision model: {e}")
                    import traceback; traceback.print_exc()
                    self.vision_model = None
                    has_vision = False
            else:
                print("✅ Vision model already compiled, reusing existing model")

            # Only process vision if compilation succeeded
            if has_vision and self.vision_model is not None:
                ctx = next(ctx for ctx in context_batch if ctx.pixel_values)
                import numpy as np
                from max.driver import Tensor, CPU
                from PIL import Image

                # Resize to required 896×896 resolution
                pixel_array = ctx.pixel_values[0].astype(np.uint8)
                if pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
                    pil_img = Image.fromarray(pixel_array)
                else:
                    pil_img = Image.fromarray(pixel_array.transpose(1, 2, 0))
                pil_img = pil_img.resize((896, 896), Image.Resampling.LANCZOS)
                print(f"🔄 Resized image from {ctx.pixel_values[0].shape} to (896, 896, 3)")

                # Normalize and batch
                pixel_array = np.array(pil_img, dtype=np.float32)
                pixel_array = (pixel_array / 127.5) - 1.0
                pixel_array = pixel_array.transpose(2, 0, 1)[None]
                print(f"📸 Processing image tensor shape: {pixel_array.shape}")

                try:
                    device = self.devices[0]
                    pixel_tensor = Tensor.from_numpy(pixel_array).to(device)
                    vision_outputs = self.vision_model.execute(pixel_tensor)
                    vision_tokens = vision_outputs[0] if isinstance(vision_outputs, (list, tuple)) else vision_outputs
                    if hasattr(vision_tokens, 'to'):
                        vision_tokens = vision_tokens.to(CPU())
                    print(f"✅ Got vision tokens shape: {vision_tokens.shape}")

                    # Convert to numpy and strip batch dim
                    vision_np = vision_tokens.to_numpy() if hasattr(vision_tokens, 'to_numpy') else np.array(vision_tokens)
                    if vision_np.ndim == 3 and vision_np.shape[0] == 1:
                        vision_np = vision_np[0]
                    print(f"🔗 Integrating vision tokens: {vision_np.shape}")

                    # FIXED: Store in temporary variables instead of context
                    vision_features_storage = vision_np
                    num_vision_tokens_storage = vision_np.shape[0]
                    print(f"✅ Vision processing successful, stored {num_vision_tokens_storage} tokens")

                except Exception as e:
                    print(f"❌ Vision model execution failed: {e}")
                    import traceback; traceback.print_exc()
                    has_vision = False

            else:
                print("⚠️ Vision processing skipped due to compilation failure")
                has_vision = False
        else:
            print("📝 Text-only input, skipping vision processing")

        # Get text inputs from parent
        text_inputs = super().prepare_initial_token_inputs(
            context_batch, kv_cache_inputs, return_n_logits
        )

        # FIXED: Integrate vision tokens if we have them
        if has_vision and vision_features_storage is not None and getattr(text_inputs, 'tokens', None) is not None:
            import numpy as np
            from max.driver import Tensor
            print(f"🔗 Integrating {num_vision_tokens_storage} vision tokens with text")

            # Extract text token IDs
            tokens_np = text_inputs.tokens.to_numpy() if hasattr(text_inputs.tokens, 'to_numpy') else np.array(text_inputs.tokens)
            vision_ids = np.full(num_vision_tokens_storage, 32000, dtype=tokens_np.dtype)
            combined = np.concatenate([vision_ids, tokens_np])
            device_for_tokens = text_inputs.tokens.device if hasattr(text_inputs.tokens, 'device') else self.devices[0]
            text_inputs.tokens = Tensor.from_numpy(combined).to(device_for_tokens)

            # Store vision features directly in ModelInputs
            text_inputs.vision_features = vision_features_storage
            text_inputs.num_vision_tokens = num_vision_tokens_storage
            text_inputs.has_vision = True

            print(f"✅ Token integration complete:")
            print(f"   Vision tokens: {num_vision_tokens_storage}")
            print(f"   Text tokens: {len(tokens_np)}")
            print(f"   Combined tokens: {len(combined)}")
        else:
            text_inputs.has_vision = False

        print("=== MULTIMODAL PREPARE_INITIAL_TOKEN_INPUTS COMPLETED ===")
        return text_inputs



    def prepare_next_token_inputs(self, next_tokens: Tensor, prev_inputs: ModelInputs) -> ModelInputs:
        print("=== PREPARE_NEXT_TOKEN_INPUTS CALLED ===")
        logger.info("prepare_next_token_inputs called")
        return super().prepare_next_token_inputs(next_tokens, prev_inputs)

    def execute(self, model_inputs: ModelInputs, **kwargs):
        print("=== EXECUTE CALLED ===")

        # Check if this is a multimodal input with vision
        if getattr(model_inputs, "has_vision", False):
            print("🔀 Processing multimodal input with vision embeddings")
            print(f"🎯 Vision tokens: {model_inputs.num_vision_tokens}")
            print(f"📊 Total tokens: {model_inputs.tokens.shape}")
            print(f"🖼️ Vision embeddings available: {model_inputs.vision_features.shape}")

            # Generate vision summary
            vision_summary = self._generate_vision_summary(model_inputs.vision_features)
            print(f"📊 Vision summary: {vision_summary}")

            # Extract text-only tokens (skip vision tokens at start)
            tokens_np = (
                model_inputs.tokens.to_numpy()
                if hasattr(model_inputs.tokens, "to_numpy")
                else np.array(model_inputs.tokens)
            )
            text_tokens = tokens_np[model_inputs.num_vision_tokens:]
            print(f"🔤 Processing text tokens: {text_tokens.shape}")

            # FIXED: Create new ModelInputs manually instead of deepcopy
            from max.driver import Tensor
            
            device_for_tokens = (
                model_inputs.tokens.device
                if hasattr(model_inputs.tokens, "device")
                else self.devices[0]
            )
            
            # Create new ModelInputs with only the tokens changed
            text_only_inputs = ModelInputs()
            
            # Copy all non-tensor attributes safely
            for attr_name in dir(model_inputs):
                if not attr_name.startswith('_') and hasattr(model_inputs, attr_name):
                    try:
                        attr_value = getattr(model_inputs, attr_name)
                        if attr_name == 'tokens':
                            # Replace with text-only tokens
                            setattr(text_only_inputs, attr_name, 
                                Tensor.from_numpy(text_tokens).to(device_for_tokens))
                        elif hasattr(attr_value, '__call__') or isinstance(attr_value, type):
                            # Skip methods and types
                            continue
                        else:
                            # Copy regular attributes
                            setattr(text_only_inputs, attr_name, attr_value)
                    except (AttributeError, TypeError):
                        # Skip attributes that can't be copied
                        continue

            # Add vision metadata
            text_only_inputs._vision_processed = True
            text_only_inputs._vision_summary = vision_summary
            text_only_inputs._num_vision_patches = model_inputs.num_vision_tokens

            # Execute text model on text-only inputs
            result = super().execute(text_only_inputs, **kwargs)

            print("✅ Multimodal execution complete!")
            return result

        else:
            print("🔄 Executing default path...")
            return super().execute(model_inputs, **kwargs)

    def _generate_vision_summary(self, vision_features):
        """Generate a statistical summary of vision features."""
        import numpy as np
        
        # Simple statistical summary
        mean_activation = np.mean(vision_features)
        max_activation = np.max(vision_features) 
        std_activation = np.std(vision_features)
        
        # Count high-activation patches (salient regions)
        high_activation_patches = np.sum(np.max(vision_features, axis=1) > mean_activation + std_activation)
        
        summary = (f"Vision analysis: {vision_features.shape[0]} patches processed. "
                f"Average activation: {mean_activation:.3f}, "
                f"Peak activation: {max_activation:.3f}, "
                f"Salient regions: {high_activation_patches} patches")
        
        return summary

    def _ensure_tensor(self, data, device=None):
        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif isinstance(data, list):
            tensor = torch.tensor(data, dtype=torch.long)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        if device:
            tensor = tensor.to(device)
        if tensor.dtype != torch.int64:
            tensor = tensor.long()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate max sequence length using text config."""
        text_config = getattr(huggingface_config, "text_config", huggingface_config)
        return super().calculate_max_seq_len(pipeline_config, text_config)
    
    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Get KV cache parameters using text config."""
        text_config = getattr(huggingface_config, "text_config", huggingface_config)
        return super().get_kv_params(text_config, n_devices, kv_cache_config, cache_dtype)
    
    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        """Get number of layers using text config."""
        text_config = getattr(huggingface_config, "text_config", huggingface_config)
        return super().get_num_layers(text_config)
    
    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimate KV cache size using text config."""
        text_config = getattr(huggingface_config, "text_config", huggingface_config)
        return super().estimate_kv_cache_size(
            pipeline_config,
            available_cache_memory,
            devices,
            text_config,
            kv_cache_config,
            cache_dtype,
        )

# FINAL COMPLETE MODEL.PY - CPU return_n_logits + proper dtype support
# This fixes the CPU device requirement and ensures proper dtype usage for Gemma3

from __future__ import annotations

import logging
from typing import Optional, Sequence, cast, Any
import numpy as np

from max.driver import Tensor, Device, CPU
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.graph import TensorValue, DeviceRef, Graph, TensorType
from max.nn import ReturnLogits, Signals
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheInputs,
    KVCacheManager,
    estimate_kv_cache_size as _estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    SupportedEncoding,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)
from transformers import AutoConfig

# Import base Gemma3 pipeline model to extend
from ..gemma3.model import Gemma3Model

logger = logging.getLogger("max.pipelines")


class ConfigProxy:
    """Flatten HuggingFace Gemma3 nested text_config onto top-level with safe fallbacks."""

    def __init__(self, original_config: AutoConfig):
        self._original = original_config
        self._text_config = getattr(original_config, "text_config", None) or original_config
        self._flatten_config()

    def _flatten_config(self):
        # Copy original attributes
        for key in dir(self._original):
            if key.startswith("_"):
                continue
            try:
                val = getattr(self._original, key)
                if not callable(val):
                    setattr(self, key, val)
            except Exception:
                pass

        # Overlay text_config attributes
        for key in dir(self._text_config):
            if key.startswith("_"):
                continue
            try:
                val = getattr(self._text_config, key)
                if not callable(val):
                    setattr(self, key, val)
            except Exception:
                pass

        # Ensure required attributes exist with defaults
        required = {
            "num_hidden_layers": 62,
            "num_key_value_heads": 16,
            "num_attention_heads": 32,
            "head_dim": None,  # derived if missing
            "hidden_size": 5376,
            "max_position_embeddings": 131072,
            "sliding_window": 4096,
            "attention_bias": False,
            "vocab_size": 262208,
            "intermediate_size": 21504,
            "hidden_activation": "gelu_tanh",
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "rope_theta": 10000.0,
            "query_pre_attn_scalar": 224,
            "final_logit_softcapping": 30.0,
            "attn_logit_softcapping": 50.0,
            "rope_scaling": None,
            "rope_local_base_freq": 10000.0,
            "sliding_window_pattern": 1,
            "model_type": "Gemma3ForConditionalGeneration",
            # Dtype-related attributes for proper model configuration
            "torch_dtype": "bfloat16",  # Gemma3-4B default
            "use_cache": True,
        }
        for k, v in required.items():
            if not hasattr(self, k) or getattr(self, k) is None:
                if k == "head_dim":
                    hs = getattr(self, "hidden_size", 5376)
                    nh = getattr(self, "num_attention_heads", 32)
                    setattr(self, k, hs // nh if nh else 168)
                else:
                    setattr(self, k, v)

    def __getattr__(self, name: str):
        if self._text_config and hasattr(self._text_config, name):
            return getattr(self._text_config, name)
        if hasattr(self._original, name):
            return getattr(self._original, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")


class Gemma3MultimodalInputs(ModelInputs):
    """Inputs wrapper including KV cache, signal buffers, and optional vision context."""

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets: Tensor | list[Tensor],
        return_n_logits: Tensor,
        signal_buffers: list[Tensor],
        kv_cache_inputs: KVCacheInputs | None,
        vision_features: Optional[Tensor] = None,
        vision_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self.signal_buffers = signal_buffers
        self.kv_cache_inputs = kv_cache_inputs
        self.vision_features = vision_features
        self.vision_tokens = vision_tokens or 0


class Gemma3_MultiModalModel(Gemma3Model):
    """Gemma3 multimodal pipeline model with static and runtime fixes for MAX."""

    model: Model
    _strict_state_dict_loading = False

    # ---------------------------
    # Static overrides: memory estimation phase
    # ---------------------------

    @staticmethod
    def calculate_max_seq_len(pipeline_config: PipelineConfig, huggingface_config: AutoConfig) -> int:
        if pipeline_config.max_length:
            return pipeline_config.max_length
        if hasattr(huggingface_config, "text_config") and huggingface_config.text_config:
            return getattr(huggingface_config.text_config, "max_position_embeddings", 8192)
        if hasattr(huggingface_config, "max_position_embeddings"):
            return huggingface_config.max_position_embeddings
        return 8192

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        if hasattr(huggingface_config, "text_config") and huggingface_config.text_config:
            return huggingface_config.text_config.num_hidden_layers
        if hasattr(huggingface_config, "num_hidden_layers"):
            return huggingface_config.num_hidden_layers
        return 62

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig, n_devices: int, kv_cache_config: KVCacheConfig, cache_dtype: DType
    ) -> KVCacheParams:
        cfg = huggingface_config
        txt = getattr(cfg, "text_config", None) or cfg
        head_dim = getattr(txt, "head_dim", None)
        if head_dim is None:
            hs = getattr(txt, "hidden_size", 5376)
            nh = getattr(txt, "num_attention_heads", 32)
            head_dim = hs // nh if nh else 168
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=getattr(txt, "num_key_value_heads", 16),
            head_dim=head_dim,
            page_size=kv_cache_config.kv_cache_page_size,
            cache_strategy=kv_cache_config.cache_strategy,
            enable_prefix_caching=kv_cache_config.enable_prefix_caching,
            enable_kvcache_swapping_to_host=kv_cache_config.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=kv_cache_config.host_kvcache_swap_space_gb,
            n_devices=n_devices,
        )

    @staticmethod
    def estimate_kv_cache_size(
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: list[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        return _estimate_kv_cache_size(
            params=Gemma3_MultiModalModel.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=Gemma3_MultiModalModel.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            num_layers=Gemma3_MultiModalModel.get_num_layers(huggingface_config=huggingface_config),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    # ---------------------------
    # Init: runtime configuration proxy and device tracking
    # ---------------------------

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
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        logger.info("🔧 Creating device-consistent multimodal model...")
        self.target_devices = devices
        self.primary_device = devices[0]
        
        # Log device and dtype information
        logger.info(f"🎯 Primary device: {self.primary_device}")
        logger.info(f"📊 All devices: {[str(d) for d in devices]}")
        logger.info(f"🔢 Encoding dtype: {encoding.dtype}")
        logger.info(f"💾 Cache dtype: {encoding.cache_dtype}")

        proxy_config = ConfigProxy(huggingface_config)
        hf_quant = getattr(huggingface_config, "quantization_config", None)
        if hf_quant and not hasattr(proxy_config, "quantization_config"):
            proxy_config.quantization_config = hf_quant

        # Log dtype configuration
        torch_dtype = getattr(proxy_config, "torch_dtype", None)
        logger.info(f"🏷️ HF config torch_dtype: {torch_dtype}")
        
        super().__init__(
            pipeline_config,
            session,
            proxy_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.original_huggingface_config = huggingface_config
        self.vision_enabled = False  # keep disabled until vision path is validated
        self.vision_tower = None
        self.multi_modal_projector = None
        self.image_token_id = getattr(proxy_config, "image_token_id", 128256)
        logger.info("✅ Gemma3 multimodal model initialized")

    # ---------------------------
    # Device-consistent _build_graph()
    # ---------------------------

    def _build_graph(self):
        primary_device = self.primary_device
        device_ref = DeviceRef.from_device(primary_device)
        logger.info(f"🏗️ Building graph with primary device: {primary_device}")
        logger.info(f"📐 Graph device reference: {device_ref}")

        tokens_type = TensorType(DType.int64, shape=["total_seq_len"], device=device_ref)
        input_row_offsets_types = [
            TensorType(DType.uint32, shape=["input_row_offsets_len"], device=DeviceRef.from_device(device))
            for device in self.devices
        ]
        
        # CRITICAL FIX: return_n_logits MUST be on CPU as per MAX requirements
        return_n_logits_type = TensorType(DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU())
        logger.info("📌 return_n_logits set to CPU device as required by MAX")
        
        signals = Signals(devices=(DeviceRef.from_device(d) for d in self.devices))

        huggingface_config = self.huggingface_config
        if self.adapter:
            state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {k: v.data() for k, v in self.weights.items()}

        # Import from parent gemma3 package (correct path)
        from ..gemma3.model_config import Gemma3Config
        from ..gemma3.gemma3 import Gemma3

        model_config = Gemma3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            attention_bias=getattr(huggingface_config, "attention_bias", False),
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        logger.info(f"🏗️ Model config dtype: {model_config.dtype}")
        logger.info(f"🗂️ Model config cache_dtype: {model_config.kv_params.dtype}")

        nn_model = Gemma3(model_config)
        nn_model.load_state_dict(state_dict, weight_alignment=1, strict=self._strict_state_dict_loading)
        self.state_dict = nn_model.state_dict(auto_initialize=False)

        kv_inputs = self.kv_manager.input_symbols()
        flattened_kv_types = [kv for sub in kv_inputs for kv in sub]

        with Graph(
            getattr(self.huggingface_config, "model_type", "Gemma3"),
            input_types=[tokens_type, return_n_logits_type, *input_row_offsets_types, *signals.input_types(), *flattened_kv_types],
        ) as graph:
            tokens, return_n_logits, *variadic_args = graph.inputs

            input_row_offsets = [v.tensor for v in variadic_args[: len(self.devices)]]
            variadic_args = variadic_args[len(self.devices) :]
            signal_buffers = [v.buffer for v in variadic_args[: len(self.devices)]]
            variadic_args = variadic_args[len(self.devices) :]
            kv_cache = [v.tensor for v in variadic_args]

            outputs = nn_model(
                tokens=tokens.tensor,
                signal_buffers=signal_buffers,
                kv_cache_inputs_per_dev=self._unflatten_kv_inputs(kv_cache),
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=input_row_offsets,
            )
            graph.output(*outputs)

        logger.info("✅ Graph built with consistent device specifications")
        return graph

    # ---------------------------
    # KV manager and unflatten overrides to use our kv params
    # ---------------------------

    def load_kv_manager(self, session: InferenceSession, available_cache_memory: int | None) -> KVCacheManager[TextContext]:
        return load_kv_manager(
            params=self.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(self.pipeline_config, huggingface_config=self.huggingface_config),
            num_layers=self.get_num_layers(huggingface_config=self.huggingface_config),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    def _unflatten_kv_inputs(self, kv_inputs_flat: Sequence[TensorValue]) -> list[tuple[TensorValue, ...]]:
        kv_params = self.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = self.kv_manager.input_symbols()[0]
        tuple_len = len(list(fetch_types))
        return [
            tuple(kv_inputs_flat[i * tuple_len : (i + 1) * tuple_len])
            for i in range(n_devices)
        ]

    # ---------------------------
    # Input preparation with FIXED device consistency
    # ---------------------------

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        all_tokens = [ctx.tokens for ctx in context_batch]
        tokens_np = np.concatenate(all_tokens)
        offsets = np.cumsum([0] + [len(t) for t in all_tokens], dtype=np.uint32)

        primary_device = self.primary_device
        
        # Create tensors with proper device placement
        tokens_t = Tensor.from_numpy(tokens_np).to(primary_device)
        
        # CRITICAL FIX: return_n_logits must be on CPU (MAX requirement)
        cpu_device = CPU()
        logits_t = Tensor.from_numpy(np.array([return_n_logits], dtype=np.int64)).to(cpu_device)
        logger.debug(f"📌 return_n_logits placed on: {logits_t.device} (required: CPU)")

        # one offsets tensor per device (distributed path)
        input_row_offsets_tensors = [Tensor.from_numpy(offsets).to(device) for device in self.devices]

        # ensure signal buffers live on matching devices
        signal_buffers_corrected = []
        for i, buf in enumerate(self.signal_buffers):
            tgt = self.devices[i] if i < len(self.devices) else self.primary_device
            signal_buffers_corrected.append(buf if buf.device == tgt else buf.to(tgt))

        # Log device placement for debugging
        logger.debug(f"🔍 Device placement:")
        logger.debug(f"  - tokens: {tokens_t.device}")
        logger.debug(f"  - return_n_logits: {logits_t.device}")
        logger.debug(f"  - offsets: {[t.device for t in input_row_offsets_tensors]}")
        logger.debug(f"  - signals: {[buf.device for buf in signal_buffers_corrected]}")

        return Gemma3MultimodalInputs(
            tokens=tokens_t,
            input_row_offsets=input_row_offsets_tensors,
            return_n_logits=logits_t,  # CPU tensor as required
            signal_buffers=signal_buffers_corrected,
            kv_cache_inputs=kv_cache_inputs,
            vision_features=None,
            vision_tokens=0,
        )

    def prepare_next_token_inputs(self, next_tokens: Tensor, prev_model_inputs: ModelInputs) -> ModelInputs:
        prev = cast(Gemma3MultimodalInputs, prev_model_inputs)
        if next_tokens.device != self.primary_device:
            next_tokens = next_tokens.to(self.primary_device)
        return Gemma3MultimodalInputs(
            tokens=next_tokens,
            input_row_offsets=prev.input_row_offsets,
            return_n_logits=prev.return_n_logits,  # Keep on CPU
            signal_buffers=prev.signal_buffers,
            kv_cache_inputs=prev.kv_cache_inputs,
            vision_features=None,
            vision_tokens=prev.vision_tokens,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        mm_inputs = cast(Gemma3MultimodalInputs, model_inputs)
        
        # Enhanced device debugging
        logger.debug(f"🔍 Execute device check:")
        logger.debug(f"  - Primary device: {self.primary_device}")
        logger.debug(f"  - tokens: {mm_inputs.tokens.device}")
        logger.debug(f"  - return_n_logits: {mm_inputs.return_n_logits.device}")
        if isinstance(mm_inputs.input_row_offsets, list):
            logger.debug(f"  - offsets: {[t.device for t in mm_inputs.input_row_offsets]}")
        else:
            logger.debug(f"  - offsets: {mm_inputs.input_row_offsets.device}")
        logger.debug(f"  - signals: {[buf.device for buf in mm_inputs.signal_buffers]}")
        
        try:
            result = super().execute(model_inputs)
            logger.debug("✅ Model execution completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Model execution failed: {e}")
            logger.error("💡 Check device placement - return_n_logits must be on CPU")
            raise
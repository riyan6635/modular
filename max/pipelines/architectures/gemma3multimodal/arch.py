from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)
from . import weight_adapters
from .model import Gemma3MultimodalModel
from .tokenizer import Gemma3MultimodalTokenizer

gemma3_multimodal_arch = SupportedArchitecture(
    name="Gemma3ForConditionalGeneration",  # Must match HuggingFace model class name
    example_repo_ids=[
        # "/home/cognida/riyan/models/gemma-3-4b-it",
        "google/gemma-3-4b-it", 
        "google/gemma-3-12b-it", 
        "google/gemma-3-27b-it",
    ],
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
        SupportedEncoding.float8_e4m3fn: [KVCacheStrategy.PAGED],
    },
    pipeline_model=Gemma3MultimodalModel,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=Gemma3MultimodalTokenizer,  # Use built-in TextTokenizer
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)
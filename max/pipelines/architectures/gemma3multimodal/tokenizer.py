from __future__ import annotations
import io
import functools
from typing import Union, Sequence, Any
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

# Correct imports based on actual codebase
from max.interfaces import (
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib.tokenizer import (
    PipelineTokenizer,
    run_with_default_executor,
    max_tokens_to_generate,
)

from .feature_extractor import Gemma3FeatureExtractor

class Gemma3MultimodalTokenizer(
    PipelineTokenizer[
        TextAndVisionContext,
        np.ndarray[np.integer[Any]],
        TextGenerationRequest,
    ]
):
    """Custom tokenizer for Gemma3 multimodal (images + text)."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        
        # Initialize text tokenizer only
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length
        
        # Setup encoding functions
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )
        
        # Initialize our custom feature extractor
        self.feature_extractor = Gemma3FeatureExtractor()
        
        # Cache EOS token IDs
        self._default_eos_token_ids = {self.eos}

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return True

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        """Apply chat template for Gemma3."""
        # Use the delegate's chat template
        templated_message = self.delegate.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(templated_message, str)
        return templated_message

    async def encode(
        self, prompt: Union[str, Sequence[int]], add_special_tokens: bool = True
    ) -> np.ndarray[np.integer[Any]]:
        """Transform the provided prompt into a token array."""
        encoded_prompt: np.ndarray[np.integer[Any]]
        
        if isinstance(prompt, str):
            if add_special_tokens:
                encoded_prompt = await run_with_default_executor(
                    self._encode_with_special_tokens, prompt
                )
            else:
                encoded_prompt = await run_with_default_executor(
                    self._encode_without_special_tokens, prompt
                )
            
            if self.max_length and len(encoded_prompt) > self.max_length:
                raise ValueError(
                    f"Input string is larger than tokenizer's max length "
                    f"({len(encoded_prompt)} > {self.max_length})."
                )
            encoded_prompt = np.array(encoded_prompt)
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    async def decode(
        self, encoded: np.ndarray[np.integer[Any]], **kwargs
    ) -> str:
        """Transform a provided encoded token array, back into readable text."""
        return self.delegate.decode(encoded, **kwargs)

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext object for Gemma3 multimodal."""
        
        # Build prompt from messages or use direct prompt
        if request.messages:
            prompt = self.apply_chat_template(request.messages)
            add_special = False
        elif request.prompt:
            prompt = request.prompt
            add_special = True
        else:
            raise ValueError("No messages or prompt provided")

        # Encode text tokens
        token_ids = await self.encode(prompt, add_special_tokens=add_special)

        # Process images using our custom feature extractor
        pixel_values = tuple()
        if request.images:
            processed_images = []
            for img_bytes in request.images:
                # Use our feature extractor instead of HF processor
                img_tensor = self.feature_extractor(img_bytes)
                processed_images.append(img_tensor)
            
            if processed_images:
                # Concatenate all images into a batch
                pixel_values = (np.concatenate(processed_images, axis=0),)

        # Calculate max generation tokens
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            token_ids.shape[0], self.max_length, max_new_tokens
        )

        # Handle EOS tokens
        if request.sampling_params.ignore_eos:
            eos_token_ids = set()
        else:
            eos_token_ids = self._default_eos_token_ids

        # Create context with processed data
        context = TextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            pixel_values=pixel_values,
            extra_model_args={},  # No aspect ratio info needed for our implementation
            tokens=token_ids,
            max_length=(
                token_ids.shape[0] + max_gen_tokens
                if max_gen_tokens is not None
                else self.max_length
            ),
            json_schema=None,
            sampling_params=request.sampling_params,
        )
        return context

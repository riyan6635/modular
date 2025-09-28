from __future__ import annotations
from typing import Sequence, Optional, Union, Tuple
import numpy as np
from PIL import Image
import io
from max.interfaces import PipelineTokenizer, TextGenerationRequest
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import TextTokenizer


class Gemma3MultimodalTokenizer(PipelineTokenizer[TextAndVisionContext, np.ndarray, TextGenerationRequest]):
    """
    Tokenizer that handles both text prompts and image arrays, producing
    a TextAndVisionContext for multimodal pipelines.
    """
    def __init__(self, model_path: Optional[str] = "google/gemma-3-4b-it", **kwargs):
        # Delegate text-only tokenization to built-in TextTokenizer
        super().__init__()
        self.text_tokenizer = TextTokenizer(model_path=model_path, **kwargs)

    @property
    def eos(self) -> int:
        return getattr(self.text_tokenizer, 'eos', 1)  # Provide default if missing

    @property
    def bos(self) -> int:
        return getattr(self.text_tokenizer, 'bos', 0)

    @property
    def pad(self) -> int:
        return getattr(self.text_tokenizer, 'pad', 0)

    @property
    def unk(self) -> int:
        return getattr(self.text_tokenizer, 'unk', 0)


    async def encode(
        self,
        prompt: Union[str, Sequence[int]],
        add_special_tokens: bool = True
    ) -> np.ndarray:
        # Use TextTokenizer.encode (sync) as delegate
        tokens = await self.text_tokenizer.encode(prompt, add_special_tokens)
        return tokens

    async def decode(self, encoded: np.ndarray, **kwargs) -> str:
        return await self.text_tokenizer.decode(encoded, **kwargs)

    async def new_context(
        self,
        request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """
        Create a multimodal context including text tokens and optional images.
        Expects request.prompt as text and request.images as list of numpy arrays.
        """
        # Encode text prompt
        text_prompt = request.prompt
        if not text_prompt and request.messages:
            # Concatenate texts from all message parts of type 'text'
            text_parts = []
            for message in request.messages:
                content = message.get("content")
                if isinstance(content, list):
                    # content is list of multimodal objects
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                elif isinstance(content, str):
                    text_parts.append(content)
            text_prompt = " ".join(text_parts).strip()

        if not text_prompt:
            raise ValueError("No prompt provided in request")

        # Then proceed with encoding text_prompt
        tokens = await self.encode(text_prompt, add_special_tokens=True)

        # Prepare pixel_values as before from request.images ...
        pixel_values: Tuple[np.ndarray, ...] = ()
        print("In tokenizer new_context")
        if hasattr(request, "images") and request.images:
            arrays = []
            for i, img in enumerate(request.images):
                if isinstance(img, np.ndarray):
                    arrays.append(img)
                elif isinstance(img, bytes):
                    # Convert bytes to PIL image then numpy array
                    pil_img = Image.open(io.BytesIO(img)).convert('RGB')  # Ensure 3 channels
                    np_img = np.array(pil_img)
                    arrays.append(np_img)
                else:
                    raise TypeError(f"Unsupported image type: {type(img)}")
                save_path = f"/home/cognida/riyan/modular/max/pipelines/architectures/gemma3multimodal/received_image_{i}.png"
                pil_img = Image.fromarray(np_img)
                pil_img.save(save_path)
                print(f"Saved received image to {save_path}")
            pixel_values = tuple(arrays)
            print(f"Received {len(arrays)} images, shapes: {[arr.shape for arr in arrays]}")
        # logger.info(f"Received {len(arrays)} images, shapes: {[arr.shape for arr in arrays]}")
        
        # Build context
        ctx = TextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids={self.text_tokenizer.eos},
            tokens=np.array(tokens, dtype=np.int64),
            max_length=len(tokens) + 512,  # allow additional tokens
            pixel_values=pixel_values,
            extra_model_args={},
            json_schema=None
        )
        print("ctx", ctx)
        ctx.assign_to_cache(request.index)
        return ctx

    def apply_chat_template(self, messages, **kwargs) -> str:
        # Not used for multimodal pipeline
        raise NotImplementedError("Chat templating not supported for multimodal tokenizer")
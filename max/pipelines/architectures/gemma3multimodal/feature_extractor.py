# FIXED feature_extractor.py - Remove pipeline_dtype error
from __future__ import annotations

import io
import numpy as np
from PIL import Image
from typing import Tuple
from max.dtype import DType

class Gemma3FeatureExtractor:
    """Feature extractor for Gemma3 multimodal vision inputs - FIXED VERSION"""
    
    def __init__(self, image_size: int = 896):
        self.image_size = image_size
        # SigLIP normalization values
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])
    
    def __call__(self, image_bytes: bytes) -> np.ndarray:
        """Process image bytes into model input format.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            numpy array of shape (1, 3, 896, 896) ready for model input
        """
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply pan-and-scan resize
        img = self.pan_and_scan_resize(img, (self.image_size, self.image_size))
        
        # Convert to numpy array and normalize
        arr = np.array(img).astype(np.float32) / 255.0
        
        # Apply SigLIP normalization
        arr = (arr - self.mean) / self.std
        
        # FIXED: Remove the pipeline_dtype reference that was causing errors
        # Just use float32 for now
        arr = arr.astype(np.float32)
        
        # Transpose to (C, H, W) format
        arr = np.transpose(arr, (2, 0, 1))
        
        # Add batch dimension: (1, 3, H, W)
        return arr[np.newaxis]
    
    def pan_and_scan_resize(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image using pan-and-scan to preserve aspect ratio.
        
        Args:
            img: PIL Image
            target_size: (width, height) tuple
            
        Returns:
            Resized PIL Image
        """
        target_w, target_h = target_size
        original_w, original_h = img.size
        
        # Calculate scaling factors
        scale_w = target_w / original_w
        scale_h = target_h / original_h
        
        # Use the larger scale to ensure the image covers the entire target area
        scale = max(scale_w, scale_h)
        
        # Calculate new dimensions
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize image
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Calculate crop coordinates to center the image
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        # Crop to target size
        img = img.crop((left, top, right, bottom))
        
        return img
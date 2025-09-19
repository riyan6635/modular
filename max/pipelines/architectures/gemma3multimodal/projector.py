# Corrected projector.py for Gemma3 Multimodal MAX Pipeline
import max.nn as nn
from max.graph import ops
from max.dtype import DType

class MultiModalProjector(nn.Module):
    """Projects vision features to language model space - Corrected Version"""
    
    def __init__(self, vision_hidden_size, text_hidden_size):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        
        # Average pooling for dimensionality reduction
        # Note: Assuming AvgPool2d exists in MAX, otherwise implement custom pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        # RMS normalization
        self.mm_soft_emb_norm = RMSNorm(vision_hidden_size)
        
        # Linear projection if dimensions differ
        if vision_hidden_size != text_hidden_size:
            self.linear_projection = nn.Linear(
                in_features=vision_hidden_size,
                out_features=text_hidden_size,
                has_bias=False
            )
        else:
            self.linear_projection = None
    
    def __call__(self, vision_features):
        """Project vision features to text space"""
        # Apply RMS normalization
        vision_features = self.mm_soft_emb_norm(vision_features)
        
        # Reshape for pooling (batch, seq_len, hidden) -> (batch, hidden, h, w)
        batch_size = vision_features.shape[0]
        seq_len = vision_features.shape[1]
        hidden_size = vision_features.shape[2]
        
        # Assume square image patches
        side_len = int(seq_len ** 0.5)
        
        # Reshape to spatial format for pooling
        # (batch, seq_len, hidden) -> (batch, hidden, side_len, side_len)
        vision_features = ops.reshape(
            vision_features, 
            (batch_size, side_len, side_len, hidden_size)
        )
        # Transpose to (batch, hidden, side_len, side_len) for pooling
        vision_features = ops.transpose(vision_features, 0, 3, 1, 2)
        
        # Apply average pooling
        pooled_features = self.avg_pool(vision_features)
        
        # Reshape back to sequence format
        pooled_h = pooled_features.shape[2]
        pooled_w = pooled_features.shape[3]
        new_seq_len = pooled_h * pooled_w
        
        # (batch, hidden, pooled_h, pooled_w) -> (batch, new_seq_len, hidden)
        pooled_features = ops.transpose(pooled_features, 0, 2, 3, 1)
        pooled_features = ops.reshape(
            pooled_features, 
            (batch_size, new_seq_len, hidden_size)
        )
        
        # Apply linear projection if needed
        if self.linear_projection is not None:
            pooled_features = self.linear_projection(pooled_features)
        
        return pooled_features


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - Corrected Version"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        
        # Initialize weight parameter properly
        import numpy as np
        weight_data = np.ones(hidden_size, dtype=np.float32)
        self.weight = nn.Parameter(
            data=weight_data,
            dtype=DType.float32
        )
    
    def __call__(self, x):
        """Apply RMS normalization"""
        # Calculate variance along the last dimension
        x_squared = ops.pow(x, 2.0)
        variance = ops.mean(x_squared, dim=-1, keepdim=True)
        
        # RMS normalization
        x_normalized = x * ops.rsqrt(variance + self.eps)
        
        # Apply learned scaling
        return self.weight * x_normalized


# Alternative implementation if AvgPool2d is not available in MAX
class CustomAvgPool2d(nn.Module):
    """Custom average pooling implementation if MAX doesn't have AvgPool2d"""
    
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
    
    def __call__(self, x):
        """Apply custom average pooling"""
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        
        # Calculate output dimensions
        out_h = (height - kernel_h) // stride_h + 1
        out_w = (width - kernel_w) // stride_w + 1
        
        # Initialize output tensor
        output_shape = (batch_size, channels, out_h, out_w)
        
        # This would need to be implemented using available MAX ops
        # Placeholder for the actual pooling operation
        # In a real implementation, you'd need to use ops.unfold or similar
        # to extract patches and then compute means
        
        raise NotImplementedError("Custom pooling implementation needs MAX-specific ops")


# Corrected feature extractor
class Gemma3FeatureExtractor:
    """Feature extractor for Gemma3 multimodal vision inputs - Corrected Version"""
    
    def __init__(self, image_size=896):
        self.image_size = image_size
        # SigLIP normalization values
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])
    
    def __call__(self, image_bytes, dtype=DType.float32):
        """Process image bytes into model input format.
        
        Args:
            image_bytes: Raw image bytes
            dtype: Target dtype for the output tensor
            
        Returns:
            numpy array of shape (1, 3, 896, 896) ready for model input
        """
        from PIL import Image
        import io
        import numpy as np
        
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply pan-and-scan resize
        img = self.pan_and_scan_resize(img, (self.image_size, self.image_size))
        
        # Convert to numpy array and normalize
        arr = np.array(img).astype(np.float32) / 255.0
        
        # Apply SigLIP normalization
        arr = (arr - self.mean) / self.std
        
        # Convert to target dtype
        if dtype == DType.bfloat16:
            # Note: numpy doesn't have bfloat16, this would need proper conversion
            arr = arr.astype(np.float16)  # Approximation
        else:
            arr = arr.astype(np.float32)
        
        # Transpose to (C, H, W) format
        arr = np.transpose(arr, (2, 0, 1))
        
        # Add batch dimension: (1, 3, H, W)
        return arr[np.newaxis]
    
    def pan_and_scan_resize(self, img, target_size):
        """Resize image using pan-and-scan to preserve aspect ratio."""
        from PIL import Image
        
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
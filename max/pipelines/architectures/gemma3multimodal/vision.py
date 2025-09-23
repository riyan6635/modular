# vision.py
from max.graph import Graph, ops, TensorType, Weight, DeviceRef
from max.dtype import DType
from max.driver import Device  # For API parity; weights are placed via DeviceRef


class SigLIPVisionEncoder:
    def __init__(self, config):
        self.config = config

        # Config fallbacks
        if isinstance(config, dict):
            self.embed_dim = config.get("vision_embed_dim", 768)
            self.image_size = config.get("image_size", 896)
            self.patch_size = config.get("patch_size", 14)
            self.num_layers = config.get("num_vision_layers", config.get("num_layers", 12))
            self.num_heads = config.get("num_vision_heads", 16)
        else:
            self.embed_dim = getattr(config, "vision_embed_dim", 768)
            self.image_size = getattr(config, "image_size", 896)
            self.patch_size = getattr(config, "patch_size", 14)
            self.num_layers = getattr(config, "num_vision_layers", getattr(config, "num_layers", 12))
            self.num_heads = getattr(config, "num_vision_heads", 16)

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads

        def build_graph(self, device: Device, batch_size: int = 1) -> Graph:
        """Original encoder, but now accepts an explicit static batch_size (default 1).
        This keeps full compatibility while allowing pan/scan graphs to set batch_size > 1.
        """
        # Define input type for images: [B, H, W, C] with static batch_size
        image_type = TensorType(
            dtype=DType.float32,
            shape=(batch_size, self.image_size, self.image_size, 3),
        )

        with Graph("siglip_vision_encoder", input_types=[image_type]) as graph:
            image = graph.inputs[0]
            image.tensor.print("debug.input_image")

            # Patch embedding via conv2d: filter shape [Kh, Kw, Cin, Cout]
            patch_embed_weight = Weight(
                name="vision.patch_embed.weight",
                dtype=DType.float32,
                shape=(self.patch_size, self.patch_size, 3, self.embed_dim),
                device=DeviceRef.CPU(),
            )

            patches = ops.conv2d(
                image.tensor,
                patch_embed_weight,
                stride=(self.patch_size, self.patch_size),
                padding=(0, 0, 0, 0),
            )
            patches.print("debug.patches_nhwc")

            # Flatten spatial -> sequence: [B, H/P, W/P, C] -> [B, N, C]
            grid = self.image_size // self.patch_size
            num_patches = grid * grid
            embeddings = ops.reshape(patches, (batch_size, num_patches, self.embed_dim))
            embeddings.print("debug.embeddings_before_pos")

            # Position embeddings (same across batch)
            pos_embed_weight = Weight(
                name="vision.pos_embed",
                dtype=DType.float32,
                shape=(num_patches, self.embed_dim),
                device=DeviceRef.CPU(),
            )
            # broadcast to [B, N, E]
            pos_embed = ops.broadcast_to(pos_embed_weight, (batch_size, num_patches, self.embed_dim))
            hidden_states = ops.add(embeddings, pos_embed)
            hidden_states.print("debug.embeddings_after_pos")

            # Transformer encoder (works on batch of sequences)
            for layer_idx in range(int(self.num_layers)):
                hidden_states = self._build_encoder_layer(hidden_states, layer_idx, batch_size)

            # Final LayerNorm
            final_gamma = Weight(
                name="vision.final_ln.weight",
                dtype=DType.float32,
                shape=(self.embed_dim,),
                device=DeviceRef.CPU(),
            )
            final_beta = Weight(
                name="vision.final_ln.bias",
                dtype=DType.float32,
                shape=(self.embed_dim,),
                device=DeviceRef.CPU(),
            )
            output = ops.layer_norm(hidden_states, final_gamma, final_beta, epsilon=1e-5)
            output.print("debug.output")  # shape: [B, N, E]

            graph.output(output)

        return graph

    def _build_encoder_layer(self, hidden_states, layer_idx, batch_size=1):
        """Single encoder layer variant that uses explicit batch_size when reshaping."""
        # LayerNorm 1
        ln1_gamma = Weight(
            name=f"vision.layers.{layer_idx}.ln1.weight",
            dtype=DType.float32,
            shape=(self.embed_dim,),
            device=DeviceRef.CPU(),
        )
        ln1_beta = Weight(
            name=f"vision.layers.{layer_idx}.ln1.bias",
            dtype=DType.float32,
            shape=(self.embed_dim,),
            device=DeviceRef.CPU(),
        )
        x = ops.layer_norm(hidden_states, ln1_gamma, ln1_beta, epsilon=1e-5)
        x.print(f"debug.layer{layer_idx}.ln1")

        # QKV projections
        wq = Weight(
            name=f"vision.layers.{layer_idx}.attn.wq",
            dtype=DType.float32,
            shape=(self.embed_dim, self.embed_dim),
            device=DeviceRef.CPU(),
        )
        wk = Weight(
            name=f"vision.layers.{layer_idx}.attn.wk",
            dtype=DType.float32,
            shape=(self.embed_dim, self.embed_dim),
            device=DeviceRef.CPU(),
        )
        wv = Weight(
            name=f"vision.layers.{layer_idx}.attn.wv",
            dtype=DType.float32,
            shape=(self.embed_dim, self.embed_dim),
            device=DeviceRef.CPU(),
        )
        wo = Weight(
            name=f"vision.layers.{layer_idx}.attn.wo",
            dtype=DType.float32,
            shape=(self.embed_dim, self.embed_dim),
            device=DeviceRef.CPU(),
        )

        q = ops.matmul(x, wq)
        k = ops.matmul(x, wk)
        v = ops.matmul(x, wv)

        # Reshape to [B, N, H, D] -> [B, H, N, D]
        # Here we use batch_size provided by the caller so attention is per-crop
        q = ops.reshape(q, (batch_size, -1, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, -1, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, -1, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Attention: [B, H, N, D] @ [B, H, D, N] -> [B, H, N, N]
        kt = ops.transpose(k, (0, 1, 3, 2))
        attn_scores = ops.matmul(q, kt)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = attn_scores * ops.constant(scale, DType.float32, DeviceRef.CPU())
        attn_probs = ops.softmax(attn_scores)
        context = ops.matmul(attn_probs, v)  # [B, H, N, D]

        # Merge heads: [B, H, N, D] -> [B, N, H*D]
        context = ops.transpose(context, (0, 2, 1, 3))
        context = ops.reshape(context, (batch_size, -1, self.embed_dim))

        attn_out = ops.matmul(context, wo)
        attn_out.print(f"debug.layer{layer_idx}.attn_out")

        # Residual
        hidden_states = ops.add(hidden_states, attn_out)

        # LayerNorm 2
        ln2_gamma = Weight(
            name=f"vision.layers.{layer_idx}.ln2.weight",
            dtype=DType.float32,
            shape=(self.embed_dim,),
            device=DeviceRef.CPU(),
        )
        ln2_beta = Weight(
            name=f"vision.layers.{layer_idx}.ln2.bias",
            dtype=DType.float32,
            shape=(self.embed_dim,),
            device=DeviceRef.CPU(),
        )
        y = ops.layer_norm(hidden_states, ln2_gamma, ln2_beta, epsilon=1e-5)
        y.print(f"debug.layer{layer_idx}.ln2")

        # MLP: E -> 4E -> E
        w1 = Weight(
            name=f"vision.layers.{layer_idx}.mlp.w1",
            dtype=DType.float32,
            shape=(self.embed_dim, 4 * self.embed_dim),
            device=DeviceRef.CPU(),
        )
        b1 = Weight(
            name=f"vision.layers.{layer_idx}.mlp.b1",
            dtype=DType.float32,
            shape=(4 * self.embed_dim,),
            device=DeviceRef.CPU(),
        )
        w2 = Weight(
            name=f"vision.layers.{layer_idx}.mlp.w2",
            dtype=DType.float32,
            shape=(4 * self.embed_dim, self.embed_dim),
            device=DeviceRef.CPU(),
        )
        b2 = Weight(
            name=f"vision.layers.{layer_idx}.mlp.b2",
            dtype=DType.float32,
            shape=(self.embed_dim,),
            device=DeviceRef.CPU(),
        )

        mlp_hidden = ops.matmul(y, w1) + b1
        mlp_hidden = ops.gelu(mlp_hidden)
        mlp_out = ops.matmul(mlp_hidden, w2) + b2
        mlp_out.print(f"debug.layer{layer_idx}.mlp_out")

        # Residual
        return ops.add(hidden_states, mlp_out)


    def build_panscan_graph(self, device: Device, max_crops: int = 8, pool: str = "mean") -> Graph:
        """
        Build a pan-scan graph that accepts `max_crops` crops in the batch dimension:
          - Input shape: [max_crops, image_size, image_size, 3]
        Notes:
          - Host must provide `max_crops` crops (pad with zeros if fewer).
          - Each crop is encoded independently (attention confined per crop).
          - Per-crop pooling collapses [N_patches] -> embedding (mean pooling),
            then crop embeddings aggregated across crops (mean or max).
        Args:
          device: Device object for graph building
          max_crops: static batch size (pad host-side to this)
          pool: 'mean' or 'max' aggregation across crops
        Returns:
          Graph that outputs a single embedding tensor of shape [1, embed_dim]
        """
        # Build encoder graph with static batch_size = max_crops
        # Reuse build_graph internals by invoking it with batch_size=max_crops, but
        # to avoid creating duplicate Graph objects we inline similar logic here
        image_type = TensorType(
            dtype=DType.float32,
            shape=(max_crops, self.image_size, self.image_size, 3),
        )

        with Graph("siglip_vision_panscan", input_types=[image_type]) as graph:
            image = graph.inputs[0]
            image.tensor.print("debug.panscan.input")

            # Patch embedding (same weights as single-image graph)
            patch_embed_weight = Weight(
                name="vision.patch_embed.weight",
                dtype=DType.float32,
                shape=(self.patch_size, self.patch_size, 3, self.embed_dim),
                device=DeviceRef.CPU(),
            )

            patches = ops.conv2d(
                image.tensor,
                patch_embed_weight,
                stride=(self.patch_size, self.patch_size),
                padding=(0, 0, 0, 0),
            )
            patches.print("debug.panscan.patches_nhwc")

            grid = self.image_size // self.patch_size
            num_patches = grid * grid

            # [B, H/P, W/P, E] -> [B, N, E]
            embeddings = ops.reshape(patches, (max_crops, num_patches, self.embed_dim))
            embeddings.print("debug.panscan.embeddings_before_pos")

            pos_embed_weight = Weight(
                name="vision.pos_embed",
                dtype=DType.float32,
                shape=(num_patches, self.embed_dim),
                device=DeviceRef.CPU(),
            )
            pos_embed = ops.broadcast_to(pos_embed_weight, (max_crops, num_patches, self.embed_dim))
            hidden_states = ops.add(embeddings, pos_embed)
            hidden_states.print("debug.panscan.emb_after_pos")

            # Run transformer encoder (per-crop, because batch dimension = max_crops)
            for layer_idx in range(int(self.num_layers)):
                hidden_states = self._build_encoder_layer(hidden_states, layer_idx, batch_size=max_crops)

            # Final LayerNorm -> output shape [B, N, E]
            final_gamma = Weight(
                name="vision.final_ln.weight",
                dtype=DType.float32,
                shape=(self.embed_dim,),
                device=DeviceRef.CPU(),
            )
            final_beta = Weight(
                name="vision.final_ln.bias",
                dtype=DType.float32,
                shape=(self.embed_dim,),
                device=DeviceRef.CPU(),
            )
            output = ops.layer_norm(hidden_states, final_gamma, final_beta, epsilon=1e-5)
            output.print("debug.panscan.output")  # [B, N, E]

            # Pool patches -> per-crop embedding [B, E]
            # Use reduce_mean to collapse seq dim (axis=1) to get per-crop vector
            per_crop_vec = ops.reduce_mean(output, axes=(1,))  # expects API: reduce_mean(tensor, axes=tuple)
            per_crop_vec.print("debug.panscan.per_crop_vec")  # [B, E]

            # Aggregate across crops -> single embedding [E]
            if pool == "mean":
                aggregated = ops.reduce_mean(per_crop_vec, axes=(0,))  # mean over batch -> [E]
            elif pool == "max":
                aggregated = ops.reduce_max(per_crop_vec, axes=(0,))  # max over batch -> [E]
            else:
                raise ValueError("pool must be 'mean' or 'max'")

            # Expand dims to [1, E] so it matches expected single-embedding output shape
            final_embed = ops.reshape(aggregated, (1, self.embed_dim))
            final_embed.print("debug.panscan.final_embed")

            graph.output(final_embed)

        return graph

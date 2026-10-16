"""Image encoder + token embeddings + Transformer encoder + sum-pooled classifier.

Shape walk-through (B = batch, L = operands per example = 10, D = d_model = 128):

    images       float32 [B, L, 1, 28, 28] --select image slots--> [N_img, 1, 28, 28]
        ImageEncoder (shared CNN)                                  -> [N_img, D]
        scatter back into a [B, L, D] grid (zeros at text slots)
    token_ids    int64   [B, L]  --nn.Embedding(11, D, padding_idx=0)--> [B, L, D]
    is_image     bool    [B, L]  --torch.where picks image vs text feature per slot--> [B, L, D]
        + modality embedding (optional; 0 = text, 1 = image)                 -> [B, L, D]
    TransformerEncoderLayer x num_layers (bidirectional; padding keys masked) -> [B, L, D]
    final LayerNorm                                                          -> [B, L, D]
    masked sum pooling over L (padding contributes 0)                        -> [B, D]
    Linear head                                                              -> [B, 91] logits

No positional embeddings: an operand's position carries no information for addition, and
without positions the model is permutation-invariant (tested in tests/test_model.py).

Attention: d_model=128 is split into nhead=4 heads of 32 dims each. Each head computes its own
attention pattern over the L operands, and the heads' outputs are concatenated back to 128.
Feedforward: each layer also applies a per-operand MLP 128 -> 512 -> 128 (GELU) after
attention. Sum pooling adds the per-operand output vectors. This makes aggregation scale with
the number of operands; it does not hard-code arithmetic, since the encoder and head must still
learn features whose sum the linear head can decode.
"""

from __future__ import annotations

import torch
from torch import nn

from tokenizer import PAD_ID, VOCAB_SIZE

TEXT_MODALITY, IMAGE_MODALITY = 0, 1


class ImageEncoder(nn.Module):
    """Shared CNN: float32 [N, 1, 28, 28] -> float32 [N, d_model].

    Default (ReLU, max pooling, BatchNorm):
    conv3x3(1->32, no bias) BatchNorm ReLU maxpool2 -> [N, 32, 14, 14]
    conv3x3(32->64, no bias) BatchNorm ReLU maxpool2 -> [N, 64, 7, 7]
    flatten -> [N, 3136] -> Linear -> [N, d_model] -> LayerNorm

    - activation: "relu" (default) or "gelu" (the original encoder).
    - pooling: "max" keeps the largest of each 2x2 window (default); "avg" averages it.
    - batch_norm: normalizes each channel after its conv. It uses statistics of the images in
      the current batch during training, and running averages in eval mode. The conv bias is
      dropped because BatchNorm's learned shift replaces it. The sum model gathers only real
      image slots before the CNN, so text placeholders never enter the batch statistics.
    In a 3-seed standalone digit benchmark (CNN-encoder-test/), ReLU + max + BatchNorm reached
    0.9924 vs 0.9915 validation digit accuracy for GELU + avg without BatchNorm.

    With batch_norm=False the layer order matches the original encoder (conv, act, pool), so
    older checkpoints load unchanged. The final LayerNorm gives image features roughly unit
    per-dimension scale, matching the N(0, 1) initialization of nn.Embedding, so neither
    modality dominates the residual stream at initialization.
    """

    def __init__(
        self,
        d_model: int = 128,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        pooling: str = "max",
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        pool_layers = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}
        activations = {"relu": nn.ReLU, "gelu": nn.GELU}
        if pooling not in pool_layers:
            raise ValueError(f"pooling must be 'avg' or 'max', got {pooling!r}")
        if activation not in activations:
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")
        pool, act = pool_layers[pooling], activations[activation]

        def stage(in_channels: int, out_channels: int) -> list[nn.Module]:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            return layers + [act(), pool(2)]

        self.features = nn.Sequential(*stage(1, conv1_channels), *stage(conv1_channels, conv2_channels))
        self.projection = nn.Linear(conv2_channels * 7 * 7, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.features(images)  # [N, 64, 7, 7]
        flat = feature_maps.flatten(start_dim=1)  # [N, 3136]
        return self.norm(self.projection(flat))  # [N, D]


class DigitSumTransformer(nn.Module):
    """Classifies the sum of a mixed text/image operand sequence into 9*L + 1 classes."""

    def __init__(
        self,
        num_classes: int = 91,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        cnn_pooling: str = "max",
        cnn_activation: str = "relu",
        cnn_batch_norm: bool = True,
        use_modality_embedding: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            d_model, conv1_channels, conv2_channels, pooling=cnn_pooling, activation=cnn_activation, batch_norm=cnn_batch_norm
        )
        # Row PAD_ID is initialized to zeros and never receives gradient.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        # The modality embedding tags each slot with which encoder produced it (text or CNN), so
        # attention can treat the two feature distributions differently. Disable it to ablate.
        self.modality_embedding = nn.Embedding(2, d_model) if use_modality_embedding else None
        # Layers are constructed separately so each gets its own random init. nn.TransformerEncoder
        # instead deep-copies one layer, so all of its layers start with identical weights.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,  # tensors are [B, L, D]
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.head = nn.Linear(d_model, num_classes)
        # Zero-init the head. The pooled vector sums L LayerNorm outputs, so its norm is large, and
        # a randomly initialized head would start with large, overconfident logits (measured
        # initial loss 9.3 vs ln(91) = 4.51). With zero init, every class starts equally likely.
        # The head still learns from step one, since dL/dW = (softmax - onehot) * pooled != 0.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def embed_operands(
        self, images: torch.Tensor, token_ids: torch.Tensor, is_image: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-slot input vectors [B, L, D]: CNN features at image slots, embeddings elsewhere."""
        text_features = self.token_embedding(token_ids)  # [B, L, D]; PAD rows are zero vectors

        image_slots = is_image & ~padding_mask  # [B, L]
        image_features = torch.zeros_like(text_features)  # [B, L, D]
        num_images = int(image_slots.sum())
        if num_images > 0:  # skip the CNN entirely for all-text batches
            selected_images = images[image_slots]  # [N_img, 1, 28, 28], only real images
            encoded_images = self.image_encoder(selected_images)  # [N_img, D]
            image_features[image_slots] = encoded_images.to(image_features.dtype)

        features = torch.where(image_slots.unsqueeze(-1), image_features, text_features)  # [B, L, D]
        if self.modality_embedding is not None:
            features = features + self.modality_embedding(is_image.long())  # [B, L, D]
        return features

    def encode(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Bidirectional encoder [B, L, D] -> [B, L, D].

        There is no causal mask, so every operand attends to every other real operand.
        src_key_padding_mask=True hides padded slots as attention keys.
        """
        hidden = features
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
        return self.final_norm(hidden)

    @staticmethod
    def pool(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Masked sum over operands [B, L, D] -> [B, D].

        masked_fill (rather than multiplying by a 0/1 mask) also removes non-finite values that
        a padded slot could hold.
        """
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0).sum(dim=1)

    def forward(
        self, images: torch.Tensor, token_ids: torch.Tensor, is_image: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Inputs follow the batch contract in dataset.collate_batch; returns logits [B, num_classes].

        The signature takes only model inputs, so targets and digit labels cannot be passed in.
        """
        features = self.embed_operands(images, token_ids, is_image, padding_mask)  # [B, L, D]
        hidden = self.encode(features, padding_mask)  # [B, L, D]
        pooled = self.pool(hidden, padding_mask)  # [B, D]
        return self.head(pooled)  # [B, num_classes]


def build_model(cfg) -> DigitSumTransformer:
    """Construct the model from an ExperimentConfig."""
    m = cfg.model
    return DigitSumTransformer(
        num_classes=cfg.num_classes,
        d_model=m.d_model,
        nhead=m.nhead,
        num_layers=m.num_layers,
        dim_feedforward=m.dim_feedforward,
        dropout=m.dropout,
        conv1_channels=m.conv1_channels,
        conv2_channels=m.conv2_channels,
        cnn_pooling=m.cnn_pooling,
        cnn_activation=m.cnn_activation,
        cnn_batch_norm=m.cnn_batch_norm,
        use_modality_embedding=m.use_modality_embedding,
        norm_first=m.norm_first,
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Trainable parameter counts per top-level component, plus the total."""
    counts = {
        name: sum(p.numel() for p in child.parameters() if p.requires_grad) for name, child in model.named_children()
    }
    counts["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def model_inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Select *only* model inputs from a batch dict and move them to `device`."""
    return {key: batch[key].to(device, non_blocking=True) for key in ("images", "token_ids", "is_image", "padding_mask")}

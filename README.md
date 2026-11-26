# Hybrid 3D-CNN + Transformer for Video Fight Detection

A deep learning model built in PyTorch that detects physical altercations (fight vs. non-fight) in video clips by combining **3D Convolutional Neural Networks (3D-CNN)** for spatiotemporal feature extraction with a **Transformer Encoder** for long-range temporal modeling.

---

## Architecture Overview

The model processes raw video tensors of shape `(Batch, Channels, Frames, Height, Width)` through two main stages:

1. **Spatial-Temporal Feature Extractor (`CNN3D`)**:
   * Three sequential `Conv3d -> BatchNorm3d -> ReLU -> MaxPool3d` blocks (`3 -> 32 -> 64 -> 128` channels) extract local visual and motion features.
   * Spatial dimensions (`H, W`) are downsampled at each stage while preserving the temporal sequence length (`T`).
   * `AdaptiveAvgPool3d((None, 1, 1))` collapses spatial features into a `128`-dimensional embedding per frame.

2. **Sequence Modeling & Classification (`HybridFightDetector`)**:
   * **Linear Projection:** Maps the `128`-D frame embeddings to `d_model` (default: `256`).
   * **Positional Encoding:** Adds learnable temporal position embeddings so the Transformer retains frame order.
   * **Transformer Encoder:** Uses Pre-LayerNorm multi-head self-attention (`num_heads=4`, `num_layers=2`) to capture interactions across all frames in the clip.
   * **Classification Head:** Applies global average pooling across the temporal dimension followed by `LayerNorm`, `Dropout`, and a linear output layer for class predictions.

---

## Project Structure

```text
├── model.py          # 3D-CNN backbone and HybridFightDetector model definition
└── README.md         # Project documentation

"""Default training configuration for the chess neural Layer 7."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NeuralConfig:
    """Training configuration."""

    # Encoder
    encoder_output_dim: int = 384
    candidate_dim: int = 48
    embed_dim: int = 8

    # Bridge
    bridge_dim: int = 128

    # Decoder
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 2
    ternary_enabled: bool = True
    partial_ternary: bool = False
    num_plan_types: int = 7
    max_candidates: int = 5

    # Loss
    margin: float = 0.3
    plan_weight: float = 0.2

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_iterations: int = 5000
    max_grad_norm: float = 1.0

    # PAB
    checkpoint_interval: int = 50

    # Data
    train_path: str = "data/chess_train.jsonl"
    eval_path: str = "data/chess_eval.jsonl"

    # Device
    device: str = "mps"

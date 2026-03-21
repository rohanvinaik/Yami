"""Training configuration with architecture variant factory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NeuralConfig:
    """Training configuration."""

    # Architecture variant
    variant: str = "A"

    # Encoder
    encoder_output_dim: int = 384
    candidate_dim: int = 48
    embed_dim: int = 8
    profile_continuous_dim: int = 3
    candidate_input_dim: int = 30

    # Bridge
    bridge_dim: int = 128

    # Decoder
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 2
    ternary_enabled: bool = False
    partial_ternary: bool = False
    num_plan_types: int = 7
    max_candidates: int = 5

    # Attention decoder (variant E)
    attention_heads: int = 4

    # Staged warmup (variant D)
    staged_warmup_steps: int = 2000

    # Loss
    margin: float = 0.3
    plan_weight: float = 0.2

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_iterations: int = 10000
    max_grad_norm: float = 1.0
    checkpoint_interval: int = 100

    # PAB
    pab_stability_threshold: float = 0.03
    pab_window: int = 10
    early_exit_patience: int = 500

    # Data
    train_path: str = "data/chess_train.jsonl"
    eval_path: str = "data/chess_eval.jsonl"

    # Device
    device: str = "cpu"

    # Campaign
    campaign_name: str = ""

    @classmethod
    def from_variant(cls, variant: str, **overrides: object) -> NeuralConfig:
        """Create a config preset for an architecture variant."""
        presets: dict[str, dict[str, object]] = {
            "A": {
                "variant": "A",
                "ternary_enabled": False,
                "partial_ternary": False,
                "decoder_hidden_dim": 256,
                "decoder_num_layers": 2,
                "encoder_output_dim": 384,
                "bridge_dim": 128,
            },
            "B": {
                "variant": "B",
                "ternary_enabled": False,
                "partial_ternary": False,
                "decoder_hidden_dim": 512,
                "decoder_num_layers": 3,
                "encoder_output_dim": 512,
                "bridge_dim": 256,
            },
            "C": {
                "variant": "C",
                "ternary_enabled": True,
                "partial_ternary": True,
                "decoder_hidden_dim": 256,
                "decoder_num_layers": 2,
            },
            "D": {
                "variant": "D",
                "ternary_enabled": False,
                "partial_ternary": False,
                "decoder_hidden_dim": 256,
                "decoder_num_layers": 2,
                "staged_warmup_steps": 2000,
            },
            "E": {
                "variant": "E",
                "ternary_enabled": False,
                "partial_ternary": False,
                "decoder_hidden_dim": 256,
                "decoder_num_layers": 2,
                "attention_heads": 4,
            },
        }
        base = presets.get(variant, presets["A"])
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]

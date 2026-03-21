"""Tests for the neural architecture (encoder, bridge, decoder, losses)."""

import torch

from yami.datagen.contracts import CANDIDATE_FEAT_DIM
from yami.neural.bridge import InformationBridge
from yami.neural.decoder import ChessTernaryDecoder, TernaryLinear, ternary_quantize
from yami.neural.encoder import CandidateEncoder, ChessPositionEncoder
from yami.neural.losses import ChessCompositeLoss

# --- Ternary primitives ---


def test_ternary_quantize_values():
    w = torch.randn(4, 8)
    q = ternary_quantize(w)
    unique = set(q.detach().unique().tolist())
    assert unique.issubset({-1.0, 0.0, 1.0})


def test_ternary_quantize_ste_gradient():
    w = torch.randn(4, 8, requires_grad=True)
    q = ternary_quantize(w)
    loss = q.sum()
    loss.backward()
    assert w.grad is not None
    assert w.grad.shape == w.shape


def test_ternary_linear_shape():
    layer = TernaryLinear(16, 8)
    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 8)


# --- Encoder ---


def test_candidate_encoder_shape():
    enc = CandidateEncoder(input_dim=CANDIDATE_FEAT_DIM, out_dim=48)
    x = torch.randn(4, CANDIDATE_FEAT_DIM)
    out = enc(x)
    assert out.shape == (4, 48)


def test_chess_position_encoder_shape():
    enc = ChessPositionEncoder(output_dim=384, candidate_dim=48, embed_dim=8)
    batch = 4
    profile = torch.randint(0, 3, (batch, 6))
    profile_cont = torch.randn(batch, 3)
    plan_type = torch.randint(0, 7, (batch,))
    plan_act = torch.randn(batch, 1)
    cand_feats = torch.randn(batch, 5, CANDIDATE_FEAT_DIM)
    cand_mask = torch.ones(batch, 5, dtype=torch.bool)
    cand_mask[:, 3:] = False

    out = enc(profile, plan_type, plan_act, cand_feats, cand_mask,
              profile_continuous=profile_cont)
    assert out.shape == (batch, 384)


def test_encoder_without_profile_continuous():
    """Backward compat: encoder works without profile_continuous."""
    enc = ChessPositionEncoder(output_dim=384)
    batch = 2
    profile = torch.randint(0, 3, (batch, 6))
    plan_type = torch.randint(0, 7, (batch,))
    plan_act = torch.randn(batch, 1)
    cand_feats = torch.randn(batch, 5, CANDIDATE_FEAT_DIM)
    cand_mask = torch.ones(batch, 5, dtype=torch.bool)

    out = enc(profile, plan_type, plan_act, cand_feats, cand_mask)
    assert out.shape == (batch, 384)


def test_encoder_return_candidate_encodings():
    enc = ChessPositionEncoder(output_dim=384, candidate_dim=48)
    batch = 2
    profile = torch.randint(0, 3, (batch, 6))
    plan_type = torch.randint(0, 7, (batch,))
    plan_act = torch.randn(batch, 1)
    cand_feats = torch.randn(batch, 5, CANDIDATE_FEAT_DIM)
    cand_mask = torch.ones(batch, 5, dtype=torch.bool)

    out, cand_encs = enc(profile, plan_type, plan_act, cand_feats, cand_mask,
                         return_candidate_encodings=True)
    assert out.shape == (batch, 384)
    assert cand_encs.shape == (batch, 5, 48)


def test_encoder_masked_candidates_differ():
    enc = ChessPositionEncoder(output_dim=384)
    batch = 2
    profile = torch.randint(0, 3, (batch, 6))
    profile_cont = torch.randn(batch, 3)
    plan_type = torch.randint(0, 7, (batch,))
    plan_act = torch.randn(batch, 1)
    cand_feats = torch.randn(batch, 5, CANDIDATE_FEAT_DIM)

    mask_3 = torch.tensor([[True, True, True, False, False]] * batch)
    mask_5 = torch.ones(batch, 5, dtype=torch.bool)

    out_3 = enc(profile, plan_type, plan_act, cand_feats, mask_3,
                profile_continuous=profile_cont)
    out_5 = enc(profile, plan_type, plan_act, cand_feats, mask_5,
                profile_continuous=profile_cont)
    assert not torch.allclose(out_3, out_5)


# --- Bridge ---


def test_bridge_shape():
    bridge = InformationBridge(input_dim=384, bridge_dim=128)
    x = torch.randn(4, 384)
    out = bridge(x)
    assert out.shape == (4, 128)


# --- Decoder ---


def test_decoder_output_keys():
    dec = ChessTernaryDecoder(input_dim=128, hidden_dim=64, num_layers=1)
    x = torch.randn(4, 128)
    mask = torch.ones(4, 5, dtype=torch.bool)
    out = dec(x, mask)
    assert "plan_logits" in out
    assert "candidate_logits" in out
    assert "confidence" in out


def test_decoder_shapes():
    dec = ChessTernaryDecoder(input_dim=128, hidden_dim=64, num_layers=1)
    x = torch.randn(4, 128)
    mask = torch.ones(4, 5, dtype=torch.bool)
    out = dec(x, mask)
    assert out["plan_logits"].shape == (4, 7)
    assert out["candidate_logits"].shape == (4, 5)
    assert out["confidence"].shape == (4, 1)


def test_decoder_masks_invalid_candidates():
    dec = ChessTernaryDecoder(input_dim=128, hidden_dim=64, num_layers=1)
    x = torch.randn(1, 128)
    mask = torch.tensor([[True, True, True, False, False]])
    out = dec(x, mask)
    logits = out["candidate_logits"][0]
    assert logits[3].item() == float("-inf")
    assert logits[4].item() == float("-inf")
    assert logits[0].item() != float("-inf")


def test_decoder_continuous_mode():
    dec = ChessTernaryDecoder(
        input_dim=128, hidden_dim=64, num_layers=1, ternary_enabled=False,
    )
    x = torch.randn(4, 128)
    out = dec(x)
    assert out["plan_logits"].shape == (4, 7)


# --- Losses ---


def test_composite_loss_runs():
    loss_fn = ChessCompositeLoss()
    cand_logits = torch.randn(4, 5)
    plan_logits = torch.randn(4, 7)
    cand_targets = torch.randint(0, 5, (4,))
    plan_targets = torch.randint(0, 7, (4,))

    result = loss_fn(cand_logits, plan_logits, cand_targets, plan_targets)
    assert "total" in result
    assert "candidate" in result
    assert result["total"].requires_grad


def test_composite_loss_with_margin():
    loss_fn = ChessCompositeLoss(margin=0.5)
    cand_logits = torch.randn(4, 5)
    plan_logits = torch.randn(4, 7)
    cand_targets = torch.tensor([0, 1, 2, 3])
    plan_targets = torch.randint(0, 7, (4,))
    second_targets = torch.tensor([1, 2, 3, 4])

    result = loss_fn(
        cand_logits, plan_logits, cand_targets, plan_targets, second_targets
    )
    assert result["margin"].item() >= 0


# --- Full pipeline smoke test ---


def test_full_pipeline_forward():
    """Smoke test: encoder → bridge → decoder forward pass."""
    enc = ChessPositionEncoder(output_dim=384)
    bridge = InformationBridge(input_dim=384, bridge_dim=128)
    dec = ChessTernaryDecoder(input_dim=128, hidden_dim=64, num_layers=1)

    batch = 2
    profile = torch.randint(0, 3, (batch, 6))
    profile_cont = torch.randn(batch, 3)
    plan_type = torch.randint(0, 7, (batch,))
    plan_act = torch.randn(batch, 1)
    cand_feats = torch.randn(batch, 5, CANDIDATE_FEAT_DIM)
    cand_mask = torch.ones(batch, 5, dtype=torch.bool)

    enc_out = enc(profile, plan_type, plan_act, cand_feats, cand_mask,
                  profile_continuous=profile_cont)
    bridge_out = bridge(enc_out)
    dec_out = dec(bridge_out, cand_mask)

    assert dec_out["candidate_logits"].shape == (batch, 5)

    # Verify gradient flow
    loss_fn = ChessCompositeLoss()
    targets = torch.randint(0, 5, (batch,))
    plan_targets = torch.randint(0, 7, (batch,))
    loss = loss_fn(
        dec_out["candidate_logits"], dec_out["plan_logits"],
        targets, plan_targets,
    )
    loss["total"].backward()

    assert enc.profile_proj[0].weight.grad is not None
    assert bridge.projection.weight.grad is not None

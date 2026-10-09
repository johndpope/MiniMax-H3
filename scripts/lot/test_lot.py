#!/usr/bin/env python3
"""CPU checks for the Level-of-Token adapter. No weights and no GPU.

    python3 scripts/lot/test_lot.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import LotVisualAdapter  # noqa: E402
from flow import (  # noqa: E402
    _asymflow,
    asymmetric_target,
    clean_from_velocity,
    compress_patches,
    euler_step,
    fit_procrustes,
    lot_clean_loss,
    mean_basis,
    recover_dense_velocity,
    sample_noisy,
)
from infer import integrate, sigma_grid  # noqa: E402
from h3_positions import packed_positions, sample_axis, spatial_axis, video_positions  # noqa: E402
from h3 import (  # noqa: E402
    H3_EXTENTS,
    H3_HIDDEN,
    H3_PATCH,
    H3_TOKEN_DIM,
    make_h3_adapter,
    over_wan_frame_budget,
    patchify,
    unpatchify,
)
from layout import (  # noqa: E402
    TokenRect,
    blur_radius,
    dense_layout,
    layout_from_blur,
    layout_from_detail,
    layout_from_rects,
    layout_from_regions,
    layout_from_vrs,
    shape_features,
)


def test_shape_and_centers():
    features = shape_features((1, 2, 4))
    assert torch.allclose(features, torch.tensor([1.0, 2.0, 3.0, -1.0]))
    timed = shape_features((2, 2, 2), include_time=True)
    assert timed.shape == (5,) and float(timed[-1]) == 1.0
    rect = TokenRect(3, 4, 8, 1, 2, 4)
    assert rect.center() == (3.0, 4.5, 9.5)
    layout = dense_layout(1, 1, 1)
    assert torch.equal(layout.centers(), torch.zeros(1, 3))
    assert layout.compression == 1.0


def test_partition_rules():
    detail = torch.zeros(4, 4)
    detail[0, 0] = 1
    layout = layout_from_detail(detail, {4: 0.5, 2: 0.5}, root=4)
    assert layout.count == 7
    detail_t = torch.zeros(2, 4, 4)
    detail_t[1, 0, 0] = 1
    timed = layout_from_detail(detail_t, {4: 0.5, 2: 0.5}, root=4)
    assert [rect.eh for rect in timed.rects if rect.t == 0] == [4]
    assert sum(rect.t == 1 for rect in timed.rects) == 7
    assert layout.rects[0] == TokenRect(0, 0, 0, 1, 1, 1)
    assert any(rect.eh == 2 for rect in layout.rects)
    try:
        layout_from_detail(detail, {4: 0.8, 2: 0.2}, root=4)
    except ValueError:
        pass
    else:
        raise AssertionError("decreasing thresholds should fail")
    try:
        layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2), TokenRect(0, 0, 0, 1, 1, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping rects should fail")


def test_regions_prefer_finer_level():
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:2, :2] = True
    layout = layout_from_regions(8, 8, background_level=0, regions=[(mask, 3)], root=8)
    covering = [rect for rect in layout.rects if rect.u == 0 and rect.v == 0]
    assert len(covering) == 1 and covering[0].extent == (1, 1, 1)
    assert any(rect.eh == 4 for rect in layout.rects)
    assert layout.dense_count == 64


def test_blur_and_vrs():
    depth = torch.full((8, 8), 10.0)
    depth[0, 0] = 1.0
    radius = blur_radius(depth, focal_depth=1.0, r_target=4.0)
    assert float(radius[0, 0]) < 1e-4
    assert float(radius[7, 7]) > 3.0
    layout = layout_from_blur(radius, {8: 3.0, 4: 1.0, 2: 0.2}, root=8)
    origin = next(rect for rect in layout.rects if rect.u == 0 and rect.v == 0)
    assert origin.eh == 1 and origin.ew == 1
    assert any(rect.eh == 4 for rect in layout.rects)

    flat = layout_from_vrs(torch.ones(8, 8), sensitivity=0.05, ambient=0.05, root=8)
    assert flat.count == 1 and flat.rects[0].extent == (1, 8, 8)
    spiked = torch.zeros(8, 8)
    spiked[0, 0] = 1
    busy = layout_from_vrs(spiked, sensitivity=0.01, ambient=0.01, root=8)
    assert busy.count > 1


def test_procrustes_and_mean_basis():
    basis = mean_basis(4, (1, 2, 2))
    assert basis.shape == (16, 4)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-6)
    assert torch.allclose(mean_basis(5, (1, 1, 1)), torch.eye(5))

    torch.manual_seed(0)
    dense_dim, token_dim, rows = 12, 4, 64
    guess = torch.linalg.qr(torch.randn(dense_dim, token_dim, dtype=torch.float64), mode="reduced").Q
    reference = torch.randn(rows, token_dim, dtype=torch.float64)
    dense = 2.0 * reference @ guess.T
    fitted, scale = fit_procrustes(dense.float(), reference.float())
    assert torch.allclose(fitted.double(), guess, atol=1e-4)
    assert torch.allclose(scale, torch.tensor(2.0), atol=1e-4)
    assert torch.allclose(fitted.T @ fitted, torch.eye(token_dim), atol=1e-4)


def test_velocity_recovery_matches_equation_9():
    _calibration, _mixin, path = _asymflow()
    assert path.name == "common.py" and "LakonLab" in str(path)
    basis = mean_basis(4, (1, 1, 3))
    y0 = torch.randn(2, 12)
    eps = torch.randn(2, 12)
    sigma = torch.tensor([0.2, 0.8])
    y_t = (1 - sigma)[:, None] * y0 + sigma[:, None] * eps
    predicted = asymmetric_target(y0, eps, basis)
    recovered = recover_dense_velocity(predicted, y_t, basis, sigma)
    assert torch.allclose(recovered, eps - y0, atol=1e-5)
    y0_hat = clean_from_velocity(y_t, recovered, sigma)
    assert float(lot_clean_loss(y0_hat, y0, sigma)) < 1e-8
    assert torch.allclose(euler_step(eps, eps - y0, 1.0, 0.0), y0, atol=1e-6)


def test_unit_layout_reproduces_pretrained_head():
    torch.manual_seed(1)
    batch, token_dim, hidden = 2, 4, 8
    adapter = LotVisualAdapter(token_dim, hidden, [(1, 1, 1), (1, 2, 2)])
    weight_in = torch.randn(hidden, token_dim)
    bias_in = torch.randn(hidden)
    weight_out = torch.randn(token_dim, hidden)
    bias_out = torch.randn(token_dim)
    adapter.init_from_pretrained(weight_in, weight_out, bias_in, bias_out)
    tokens = torch.randn(batch, 1, 2, 2, token_dim)
    layout = dense_layout(1, 2, 2)
    velocity = adapter(tokens, 0.4, layout, lambda hidden_states, _coords, _sigma: hidden_states)
    flat = tokens.reshape(batch, 4, token_dim)
    manual = (flat @ weight_in.T + bias_in) @ weight_out.T + bias_out
    assert torch.allclose(velocity.reshape(batch, 4, token_dim), manual, atol=1e-5)

    coarse = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    embedded, centers = adapter.embed(tokens, coarse)
    packed = tokens.reshape(batch, 1, token_dim * 4)
    compressed = compress_patches(packed, adapter.bank.basis((1, 2, 2)))
    shared = compressed @ weight_in.T + bias_in
    assert torch.allclose(embedded, shared, atol=1e-5)
    assert torch.allclose(centers, torch.tensor([[0.0, 0.5, 0.5]]))


def test_extent_scale_and_backward():
    torch.manual_seed(2)
    adapter = LotVisualAdapter(4, 8, [(1, 1, 1), (1, 2, 2)])
    guess = torch.linalg.qr(torch.randn(16, 4), mode="reduced").Q
    reference = torch.randn(32, 4)
    dense = 2.0 * reference @ guess.T
    adapter.bank.fit_((1, 2, 2), dense, reference)
    assert torch.allclose(adapter.bank.scale((1, 2, 2)), torch.tensor(2.0), atol=1e-4)
    clean = torch.randn(1, 1, 2, 2, 4)
    layout = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    scaled = adapter.bank.scale_clean(clean, layout)
    assert torch.allclose(adapter.bank.unscale(scaled, layout), clean, atol=1e-5)

    y0 = torch.randn(2, 1, 4, 4, 4)
    detail = torch.zeros(4, 4)
    detail[0, 0] = 1
    mixed = layout_from_detail(detail, {4: 0.5, 2: 0.5}, root=4)
    y_t, _noise = sample_noisy(y0, 0.5)
    velocity = adapter(y_t, 0.5, mixed, lambda states, _coords, _sigma: states)
    loss = lot_clean_loss(clean_from_velocity(y_t, velocity, 0.5), y0, 0.5)
    loss.backward()
    assert adapter.out_proj["1x1x1"].weight.grad.abs().sum() > 0
    assert adapter.out_proj["1x2x2"].weight.grad.abs().sum() > 0
    assert adapter.bank.basis((1, 2, 2)).grad is None


def test_h3_patch_geometry():
    assert H3_TOKEN_DIM == 96
    assert H3_PATCH == (1, 2, 2)
    assert H3_HIDDEN == 5376
    assert len(H3_EXTENTS) == 9
    assert over_wan_frame_budget(3073, 1)
    assert not over_wan_frame_budget(3072, 1)
    latent = torch.zeros(1, 2, 1, 2, 2)
    latent[0, 1, 0, 0, 1] = 7
    tokens = patchify(latent, patch=(1, 2, 2))
    assert tokens.shape[-1] == 8
    assert float(tokens[0, 0, 0, 0, 5]) == 7
    video = torch.randn(1, 24, 2, 4, 6)
    roundtrip = unpatchify(patchify(video))
    assert torch.equal(roundtrip, video)
    adapter = make_h3_adapter(hidden_size=32)
    assert adapter.token_dim == 96
    assert adapter.bank.basis((1, 4, 4)).shape == (16 * 96, 96)


def test_fit_extent_rebuilds_heads():
    torch.manual_seed(3)
    adapter = LotVisualAdapter(4, 8, [(1, 1, 1), (1, 2, 2)])
    weight_in = torch.randn(8, 4)
    weight_out = torch.randn(4, 8)
    bias_out = torch.randn(4)
    adapter.init_from_pretrained(weight_in, weight_out, None, bias_out)
    before = adapter.out_proj["1x2x2"].weight.detach().clone()
    guess = torch.linalg.qr(torch.randn(16, 4), mode="reduced").Q
    reference = torch.randn(32, 4)
    dense = 2.0 * reference @ guess.T
    adapter.fit_extent((1, 2, 2), dense, reference)
    expected = guess.float() @ weight_out
    assert torch.allclose(adapter.out_proj["1x2x2"].weight, expected, atol=1e-4)
    assert not torch.allclose(adapter.out_proj["1x2x2"].weight, before, atol=1e-4)
    assert torch.allclose(adapter.out_proj["1x1x1"].weight, weight_out, atol=1e-5)
    lifted = adapter.bank.basis((1, 2, 2)) @ bias_out
    assert torch.allclose(adapter.out_proj["1x2x2"].bias, lifted, atol=1e-4)


def test_h3_positions_match_base_grid():
    axis = torch.tensor([0.0, 10.0, 30.0], dtype=torch.float64)
    assert float(sample_axis(axis, 0)) == 0.0
    assert float(sample_axis(axis, 1.5)) == 20.0
    layout = dense_layout(2, 2, 3)
    positions = video_positions(layout, latent_height=4, latent_width=6, origin=5.0)
    area = math.sqrt(4 * 6)
    height = spatial_axis(4, 2, area)
    width = spatial_axis(6, 2, area)
    assert torch.allclose(positions[0, 1], height[0])
    assert torch.allclose(positions[1, 2], width[1])
    coarse = layout_from_rects(2, 2, 3, [
        TokenRect(0, 0, 0, 1, 2, 2),
        TokenRect(0, 0, 2, 1, 2, 1),
        TokenRect(1, 0, 0, 1, 2, 2),
        TokenRect(1, 0, 2, 1, 2, 1),
    ])
    coarse_pos = video_positions(coarse, 4, 6, origin=5.0)
    assert torch.allclose(coarse_pos[0, 1], 0.5 * (height[0] + height[1]))
    packed, video_start = packed_positions(layout, 4, 6, text_len=3, num_audio_latents=2)
    assert video_start == 3 + 4
    assert torch.allclose(packed[video_start:], video_positions(layout, 4, 6, origin=3.0))
    assert float(packed[0, 0]) == 0.0 and float(packed[2, 0]) == 2.0
    assert float(packed[3, 0]) == 3.0 and float(packed[4, 0]) == 4.0
    assert float(packed[5, 0]) == 3.0 and float(packed[5, 2]) != float(packed[3, 2])

    import sys as _sys
    fizgig = "/media/2TB/Fizgig/src"
    if fizgig not in _sys.path:
        _sys.path.insert(0, fizgig)
    from fizgig.minimax.model import image_position_ids

    reference = image_position_ids(3, 4, 6, num_audio_latents=2, latent_t=2)
    assert torch.allclose(packed, reference)


def test_euler_inference_recovers_clean():
    layout = dense_layout(1, 2, 2)
    clean = torch.randn(2, 1, 2, 2, 3)
    noise = torch.randn_like(clean)

    def predict(_state, _t, _layout):
        return noise - clean

    sampled = integrate(predict, noise, layout, sigma_grid(4))
    assert torch.allclose(sampled, clean, atol=1e-5)
    assert sigma_grid(4).shape == (5,)
    assert float(sigma_grid(1)[0]) == 1.0 and float(sigma_grid(1)[-1]) == 0.0


def main():
    tests = [
        test_shape_and_centers,
        test_partition_rules,
        test_regions_prefer_finer_level,
        test_blur_and_vrs,
        test_procrustes_and_mean_basis,
        test_velocity_recovery_matches_equation_9,
        test_unit_layout_reproduces_pretrained_head,
        test_extent_scale_and_backward,
        test_h3_patch_geometry,
        test_fit_extent_rebuilds_heads,
        test_h3_positions_match_base_grid,
        test_euler_inference_recovers_clean,
    ]
    for test in tests:
        test()
        print(test.__name__)
    print(f"ok {len(tests)}")


if __name__ == "__main__":
    main()

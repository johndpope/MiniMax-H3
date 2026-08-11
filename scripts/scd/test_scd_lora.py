#!/usr/bin/env python3
"""Invariants for `scd_lora`. CPU, seconds, no weights and no GPU.

Same tiny 6-block model as `test_scd_model.py`, for the same reason: injection, freezing, naming
and zero-initialisation are all shape-independent, and a test that needs the 66 GB checkpoint is a
test nobody runs. What a tiny model cannot check is the one thing the real run cares about —
whether the base is `Linear4bit` — so `test_wraps_any_linear_subclass` fakes that instead.

Needs fizgig on the path (`--fizgig-src`), which CI does not have; CI lints and byte-compiles.

Usage:
    python3 scripts/scd/test_scd_lora.py
"""

import argparse
import copy
import sys

import torch

from test_scd_model import DECODER_SOURCE, ENCODER_DEPTH, build, sample_inputs

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
def test_zero_at_init(scd, _mm, inputs):
    """The composed model is bit-identical to the frozen one until something trains.

    Not allclose. `lora_up` is zeroed, so the added term is exactly `0 @ down(x)` and any drift
    here means the adapter changed the base path — a cast, a dropout left on in eval, a merge.
    """
    from scd_lora import add_lora

    with torch.no_grad():
        before = scd.encode(**inputs)[0]
    add_lora(scd, rank=4)
    scd.eval()
    with torch.no_grad():
        after = scd.encode(**inputs)[0]
    assert torch.equal(before, after), \
        f"injection moved the frozen output by {(after - before).abs().max():.3e} — the adapter " \
        "is not zero at init, so step 0 is not the base model"


@case
def test_a_trained_adapter_changes_the_output(scd, _mm, inputs):
    """...and once `lora_up` is non-zero it does. Guards the mirror of `test_zero_at_init`:
    an adapter wired in but never called would pass that test perfectly."""
    from scd_lora import add_lora

    made = add_lora(scd, rank=4)
    scd.eval()
    with torch.no_grad():
        before = scd.encode(**inputs)[0]
        for m in made.values():
            m.lora_up.weight.normal_(0, 0.02)
        after = scd.encode(**inputs)[0]
    assert not torch.equal(before, after), \
        "writing every lora_up left the encoder output unchanged — the adapters are in the tree " \
        "but not on the forward path"


@case
def test_encoder_excludes_adaln_and_decoder_keeps_it(scd, _mm, _inputs):
    """D3's asymmetry, as a fact about the installed names rather than about the constant lists."""
    from scd_lora import add_lora

    made = add_lora(scd, rank=4)
    enc_adaln = [n for n in made if n.startswith("lora_unet_blocks_") and "adaln" in n]
    dec_adaln = [n for n in made if n.startswith("lora_unet_scddec_") and "adaln" in n]
    assert not enc_adaln, f"encoder got adaln adapters {enc_adaln}; sigma is constant there"
    assert len(dec_adaln) == len(scd.decoder_blocks), \
        f"{len(dec_adaln)} decoder adaln adapters for {len(scd.decoder_blocks)} slots"


@case
def test_names_separate_the_two_copies_of_a_shared_block(scd, _mm, _inputs):
    """Block 0 exists in both halves, and its two adapters must not collide in a state dict.

    This is the failure that looks like nothing: two entries named `lora_unet_blocks_0_...` would
    have one silently overwrite the other on save, and the loaded LoRA would apply the decoder's
    adaptation to the encoder's block.
    """
    from scd_lora import add_lora, lora_state_dict

    shared = [i for i in scd.decoder_source if i < scd.encoder_depth]
    assert shared, "fixture has no block in both halves — this case is vacuous"
    made = add_lora(scd, rank=4)
    sd = lora_state_dict(scd)

    assert len(sd) == 3 * len(made), \
        f"{len(sd)} state-dict entries for {len(made)} adapters (want 3 each); names collide"
    for i in shared:
        e = f"lora_unet_blocks_{i}_attn_qkv_proj.lora_up.weight"
        d = f"lora_unet_scddec_{i}_attn_qkv_proj.lora_up.weight"
        assert e in sd and d in sd, f"block {i}: missing {e if e not in sd else d}"

    # The decoder name must carry the SOURCE block, not the slot. Slot and source agree on the
    # front pair, so the loop above passes either way; the tail is where they diverge and where a
    # slot-named checkpoint would quietly claim to adapt blocks the composition never touched.
    for src in scd.decoder_source:
        key = f"lora_unet_scddec_{src}_attn_qkv_proj.lora_up.weight"
        assert key in sd, f"no adapter named for decoder source block {src}; names are slot-based"


@case
def test_only_adapters_train(scd, _mm, _inputs):
    """Every trainable parameter is a lora factor — including `final_layer`, which `decode_frame`
    runs and would otherwise be fine-tuned from a 12-clip set as a side effect."""
    from scd_lora import LoRALinear, add_lora, lora_parameters

    add_lora(scd, rank=4)
    want = {id(p) for m in scd.modules() if isinstance(m, LoRALinear)
            for p in (m.lora_down.weight, m.lora_up.weight)}
    got = {id(p) for p in lora_parameters(scd)}
    assert got == want, f"{len(got - want)} non-adapter parameters are trainable"
    assert not any(p.requires_grad for p in scd.base.final_layer.parameters()), \
        "final_layer is trainable"


@case
def test_gradient_reaches_both_halves(scd, _mm, inputs):
    """A loss on the decoder's output puts gradient on encoder AND decoder adapters.

    The encoder half is the one at risk: its features enter the decoder as a detachable tensor,
    and anything that detaches them (a cache, a `no_grad` around the encode) leaves 30 blocks of
    adapters receiving zero gradient for the whole run while the loss still falls.
    """
    from scd_lora import add_lora

    made = add_lora(scd, rank=4)
    for m in made.values():
        m.lora_up.weight.data.normal_(0, 0.02)

    clean = dict(inputs, t=torch.tensor([1.0]))
    enc, clean_ctx, _ = scd.encode_chunked(1, **clean)
    noisy_h, noisy_ctx, pack = scd.preamble(**inputs)
    sp = scd.spans(inputs["video_latent"], enc.shape[0])
    scd.decode_frame(enc, clean_ctx, noisy_h, noisy_ctx, sp, 1,
                     media_start=pack["media_start"]).square().mean().backward()

    for half, prefix in (("encoder", "lora_unet_blocks_"), ("decoder", "lora_unet_scddec_")):
        grads = [n for n, m in made.items()
                 if n.startswith(prefix) and m.lora_down.weight.grad is not None
                 and m.lora_down.weight.grad.abs().sum() > 0]
        assert grads, f"no {half} adapter received gradient from the decoder loss"


@case
def test_ablation_can_train_one_half(scd, _mm, _inputs):
    """`encoder_blocks=()` is D3's decoder-heavy ablation and must install nothing on the encoder."""
    from scd_lora import add_lora

    made = add_lora(scd, rank=4, encoder_blocks=())
    assert made, "decoder-only ablation installed nothing at all"
    assert not [n for n in made if n.startswith("lora_unet_blocks_")], \
        "encoder_blocks=() still wrapped encoder blocks"


@case
def test_alpha_scales_the_update(scd, _mm, _inputs):
    """`scale = alpha / rank`, and the saved `alpha` matches — a loader that recomputes the scale
    from the two must land on the same number this run trained with."""
    from scd_lora import add_lora, lora_state_dict

    made = add_lora(scd, rank=8, alpha=16)
    m = next(iter(made.values()))
    assert m.scale == 2.0, f"scale {m.scale} != alpha/rank = 2.0"
    sd = lora_state_dict(scd)
    assert float(sd[f"{m.lora_name}.alpha"]) == 16.0
    assert sd[f"{m.lora_name}.lora_down.weight"].shape[0] == 8


@case
def test_wraps_any_linear_subclass(_scd, _mm, _inputs):
    """The real base is bnb `Linear4bit`, whose `weight` is a `Params4bit` shell with no usable
    `.data`. The adapter must read only `in_features`/`out_features` and call the module.

    Stands in for bitsandbytes, which is not a dependency of this test. What it pins is that
    nothing in `LoRALinear` touches `base.weight` — a merge path added later fails here.
    """
    from scd_lora import LoRALinear

    class FakeQuantLinear(torch.nn.Linear):
        @property
        def weight(self):
            raise AttributeError("Params4bit has no plain weight")

        def forward(self, x):
            return torch.zeros(*x.shape[:-1], self.out_features, dtype=x.dtype)

    base = FakeQuantLinear(6, 4, bias=False)
    wrapped = LoRALinear(base, rank=2)
    wrapped.lora_up.weight.data.normal_(0, 1.0)
    out = wrapped(torch.randn(3, 6))
    assert out.shape == (3, 4) and out.abs().sum() > 0


@case
def test_adapter_params_stay_fp32_under_a_bf16_base(scd, _mm, _inputs):
    """`scd.to(bfloat16)` must not drag the factors down with it, and the mixed pair must still run.

    A `.to(dtype)` on the composed model is the ordinary way to get the base into bf16, and it
    would silently take the adapters too — at which point updates of ~1e-4 against weights of
    ~1e-2 fall below bf16's ulp and the LoRA appears not to learn.

    The second half is a single adapter rather than a model forward: the base's own packer decides
    what dtype reaches `video_patch_proj` and a bf16 `video_latent` does not survive it, which is
    a fact about the base's input convention and not about this wrapper. What is at risk here is
    narrower — fp32 factors fed a bf16 activation, added back to a bf16 base output.
    """
    from scd_lora import LoRALinear, add_lora

    add_lora(scd, rank=4)
    scd.to(torch.bfloat16)
    bad = [m.lora_name for m in scd.modules()
           if isinstance(m, LoRALinear) and m.lora_down.weight.dtype != torch.float32]
    assert not bad, f"{len(bad)} adapters were cast out of fp32 by .to(bfloat16), e.g. {bad[0]}"

    m = next(mod for mod in scd.modules() if isinstance(mod, LoRALinear))
    m.lora_up.weight.data.normal_(0, 0.02)
    with torch.no_grad():
        out = m(torch.randn(3, m.base.in_features, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16, f"adapter returned {out.dtype}, not the base's bfloat16"
    assert torch.isfinite(out).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fizgig-src", default="/media/2TB/Fizgig/src")
    args = ap.parse_args()

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from scd_model import MiniMaxH3SCD

    _, mm = build(args.fizgig_src)
    inputs = sample_inputs(mm)
    failures = 0
    for fn in CASES:
        # A fresh composition per case: injection mutates the module tree in place and several
        # cases write to parameters, so a shared model would make the order significant.
        stock, _ = build(args.fizgig_src)
        scd = MiniMaxH3SCD(stock, encoder_depth=ENCODER_DEPTH, decoder_source=DECODER_SOURCE)
        try:
            fn(scd, mm, copy.deepcopy(inputs))
            print(f"ok    {fn.__name__}")
        except Exception as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

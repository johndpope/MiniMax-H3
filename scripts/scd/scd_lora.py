#!/usr/bin/env python3
"""LoRA adapters for the SCD split, with the encoder/decoder asymmetry §6.4 D3 asks for.

Wrappers, not forward-patching. Fizgig's `networks/lora.py` swaps `org_module.forward` and deletes
its reference to the module, which keeps the module tree untouched — good when the adapter has to
come off again. Here the adapter never comes off, and replacing the child in the tree buys two
things the patch cannot: the LoRA parameters are inside `scd.state_dict()` and `scd.parameters()`
so an optimizer and a checkpoint see them without a second registry, and `deepcopy` of a block
takes its adapter with it. The saved key names still follow sd-scripts, so the artifact loads
where the patch-style loader expects it.

The base is NF4 in every real run (`Params4bit`), so this never touches `base.weight`: no merge
path, no `weight.data` arithmetic — the adapter calls the frozen module and adds to its output.
That also makes it indifferent to whether the target is `nn.Linear`, bnb's `Linear4bit` or
Fizgig's `ConvRotInt8Linear`; all three are `nn.Linear` subclasses and only `in_features` /
`out_features` are read.

Two asymmetries, both deliberate:

  * **Encoder excludes `adaln_proj`** (D3). The encoder runs at a single sigma=0, so its
    modulation input is one constant vector and the gradient through that projection is rank-1 —
    adapter capacity spent there buys a scaled copy of a constant. The decoder KEEPS it, because
    its two halves sit at different timesteps by construction (`decoder_frame_input`), so its
    modulation input actually varies and is the one place the split changes what that layer is
    asked to do.
  * **Decoder names are not base names.** Encoder adapters are named for the base block they
    live on, so `lora_unet_blocks_7_attn_qkv_proj` means the same thing here as in stock H3.
    Decoder adapters cannot: the re-composition holds blocks 0 and 1 TWICE, once per half, and
    naming both `blocks_0_...` would make a stock-graph loader silently stack two different
    adapters on one block. They get `lora_unet_scddec_<source>_...` instead, which is
    deliberately not a name stock H3 has.

Usage:
    from scd_lora import add_lora, lora_state_dict
    info = add_lora(scd, rank=32, alpha=32)
    opt = torch.optim.AdamW([p for p in scd.parameters() if p.requires_grad], lr=1e-4)
    ...
    save_file(lora_state_dict(scd), "scd_lora.safetensors")
"""

import math

import torch
from torch import nn

# D3's list. `mlp.fc1` is the gated pair (it emits 2 x ffn_hidden) and `fc2` the projection back.
ENCODER_TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
# Same, plus the modulation projection — see the module docstring on why only this half gets it.
DECODER_TARGETS = ENCODER_TARGETS + ("adaln_proj.linear",)


class LoRALinear(nn.Module):
    """Frozen `base(x)` plus `scale * up(down(x))`, zero at initialisation.

    `up` is zeroed rather than both factors randomised, so the composed model at step 0 is exactly
    the frozen model. That is what makes "did the adapter do anything" a decidable question — with
    both factors random the run starts from a perturbed base and every early sample is confounded
    by an offset nobody chose.

    Adapter parameters stay fp32 while the base computes in bf16. Rank-r matmuls are a rounding
    error against a 5376-wide block, and bf16 has ~3 decimal digits: an update of 1e-4 against a
    weight of 1e-2 lands below the ulp and is silently dropped, which looks exactly like a LoRA
    that will not learn. The cast happens at the boundary, so the base's dtype is unchanged.
    """

    def __init__(self, base, rank, alpha=None, dropout=0.0, lora_name=""):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.lora_name = lora_name
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        alpha = rank if alpha is None or alpha == 0 else alpha
        self.rank = rank
        self.scale = alpha / rank
        self.register_buffer("alpha", torch.tensor(float(alpha)))

        # On the base's device, not the default one. `add_lora` runs after the checkpoint is
        # already resident on the GPU, and a `.to(device)` afterwards is not a substitute: the
        # composition parks some blocks on CPU, so a blanket move would drag those back.
        dev = next((p.device for p in base.parameters()), None)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False,
                                   dtype=torch.float32, device=dev)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False,
                                 dtype=torch.float32, device=dev)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def _apply(self, fn, *args, **kwargs):
        """Let device moves through, put the factors back in fp32 after dtype casts.

        `scd.to(torch.bfloat16)` is the ordinary way to get the base into bf16 and it recurses,
        so without this the adapters go with it and the fp32 argument above is undone by the one
        line every caller writes. Reassigning `.data` keeps the same Parameter object, so an
        optimizer built before the cast still points at it.
        """
        out = super()._apply(fn, *args, **kwargs)
        for lin in (self.lora_down, self.lora_up):
            if lin.weight.dtype != torch.float32:
                lin.weight.data = lin.weight.data.float()
        return out

    def forward(self, x):
        out = self.base(x)
        h = x.to(self.lora_down.weight.dtype)
        if self.dropout is not None:
            h = self.dropout(h)
        return out + (self.lora_up(self.lora_down(h)) * self.scale).to(out.dtype)


def _replace(parent_of, dotted, make):
    """Swap the submodule at `dotted` under `parent_of` for `make(child)`. Returns the new child."""
    parts = dotted.split(".")
    holder = parent_of
    for p in parts[:-1]:
        holder = getattr(holder, p)
    child = getattr(holder, parts[-1])
    new = make(child)
    setattr(holder, parts[-1], new)
    return new


def add_lora(scd, rank=32, alpha=None, dropout=0.0,
             encoder_targets=ENCODER_TARGETS, decoder_targets=DECODER_TARGETS,
             encoder_blocks=None, decoder_slots=None):
    """Wrap both halves' target Linears and freeze everything else. Returns `{name: LoRALinear}`.

    Freezing is global-then-thaw rather than "freeze the blocks": the base also carries the patch
    projections, the time embedding and `final_layer`, and `decode_frame(velocity=True)` runs that
    last one. Leaving it trainable would train a 5376-wide output head from a 12-clip set through
    a path nothing here tests, which is not the experiment.

    `encoder_blocks` / `decoder_slots` take an iterable of indices for D3's decoder-heavy
    ablation — `decoder_slots=range(len(scd.decoder_blocks)), encoder_blocks=()` trains the
    decoder half alone. Indices are SLOTS for the decoder (position in `decoder_blocks`), while
    the NAME carries the source block, because the slot is an artifact of this composition and
    the source block is the thing that means something across runs.
    """
    for p in scd.parameters():
        p.requires_grad_(False)

    made = {}

    def attach(owner, prefix, targets):
        for dotted in targets:
            name = f"{prefix}_{dotted.replace('.', '_')}"
            try:
                new = _replace(owner, dotted,
                               lambda c: LoRALinear(c, rank, alpha, dropout, name))
            except AttributeError:
                continue                      # target absent in this config; not an error
            made[name] = new

    enc = scd.encoder_blocks
    for i in (range(len(enc)) if encoder_blocks is None else encoder_blocks):
        attach(enc[i], f"lora_unet_blocks_{i}", encoder_targets)

    dec = scd.decoder_blocks
    for s in (range(len(dec)) if decoder_slots is None else decoder_slots):
        attach(dec[s], f"lora_unet_scddec_{scd.decoder_source[s]}", decoder_targets)

    if not made:
        raise ValueError("no LoRA targets matched — check encoder_targets/decoder_targets "
                         "against the block's Linear names")
    for m in made.values():
        for p in (m.lora_down, m.lora_up):
            p.weight.requires_grad_(True)
    return made


def lora_parameters(scd):
    return [p for p in scd.parameters() if p.requires_grad]


def lora_state_dict(scd, dtype=torch.float32):
    """sd-scripts layout: `{lora_name}.lora_down.weight`, `.lora_up.weight`, `.alpha`.

    Walks the module tree rather than the dict `add_lora` returned, so a checkpoint written
    mid-run reflects what is actually installed. The tree path (`base.blocks.7.attn.qkv_proj`)
    is not the sd-scripts name and is not derived from here — the adapter carries the name the
    code that chose it gave it.
    """
    out = {}
    for _, mod in scd.named_modules():
        if not isinstance(mod, LoRALinear):
            continue
        out[f"{mod.lora_name}.lora_down.weight"] = mod.lora_down.weight.detach().to(dtype)
        out[f"{mod.lora_name}.lora_up.weight"] = mod.lora_up.weight.detach().to(dtype)
        out[f"{mod.lora_name}.alpha"] = mod.alpha.detach().to(dtype)
    return out


def lora_report(made):
    """(n_modules, n_params) — what got wrapped and how much of it trains."""
    n = sum(m.lora_down.weight.numel() + m.lora_up.weight.numel() for m in made.values())
    return len(made), n

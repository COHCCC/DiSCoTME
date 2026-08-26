# src/models/hd_encoders.py
"""
VisiumHD-specific encoders.

A *super-bin* = N sub-bins (e.g. a 2x2 block of 16um bins). On the GENE side we
do NOT average the N sub-bin expression vectors (that throws away the sub-bin
spatial structure -- VisiumHD's main advantage). Instead we:

    1. lightly encode each sub-bin (shared LayerNorm + Linear [+ 1 GELU layer]),
    2. add a learnable per-slot (2x2) positional embedding,
    3. aggregate the N sub-bins with a learnable, mask-aware head:
         - "mil"  : Gated Attention-MIL (Ilse 2018)  [default]
         - "mlp"  : fixed-order concat -> MLP  (position + full interactions)
         - "mean" : masked mean  (baseline / ablation)
    4. feed the resulting super-bin feature into the SHARED spatial-context
       (LongNet) gated fusion -- identical downstream to the regular-Visium gene
       encoder, so the model skeleton / trainer / contrastive loss are reused.

Drop-in contract (so discotme_net.py needs no change):
    forward(target_gene, context_genes, target_pos, context_pos) -> (fused, alpha)
    where, for HD,
        target_gene  : [B, N, gene_dim]
        context_genes: [B, C, N, gene_dim]
    (regular Visium passes [B, gene_dim] / [B, C, gene_dim]; a 2-D target is
    auto-promoted to N=1 so this encoder also degrades gracefully.)

Empty sub-bins (all-zero gene vector) are detected and masked out of the
aggregation. The IMAGE side is handled in preprocessing (one contiguous patch
covering the 2x2 super-bin region), so the existing image encoder is reused.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import GENE_ENCODERS
from src.hd.multiscale import SubbinRelposEncoder


# ----------------------------------------------------------------------
# Sub-bin aggregation heads (the only "new" idea; everything else is shared)
# ----------------------------------------------------------------------
class SubBinAggregator(nn.Module):
    """Aggregate N per-sub-bin features [*, N, d] -> one super-bin feature [*, d].

    Returns (feat, attn) where attn [*, N] is the per-sub-bin weight (for the
    "mil"/"mean" heads) -- useful as a sub-bin-resolution saliency map at viz.
    """

    def __init__(self, n_sub: int, d: int, agg: str = "mil", mil_hidden: int = 128):
        super().__init__()
        self.agg = agg
        self.n_sub = n_sub
        if agg == "mil":
            # Gated Attention-MIL: a_i = softmax( w^T ( tanh(V h_i) * sigmoid(U h_i) ) )
            self.V = nn.Linear(d, mil_hidden)
            self.U = nn.Linear(d, mil_hidden)
            self.w = nn.Linear(mil_hidden, 1)
        elif agg == "mlp":
            # fixed-order concat -> MLP : slot order encodes position, MLP mixes
            # all sub-bins jointly (full pairwise + higher-order interactions).
            self.mlp = nn.Sequential(
                nn.Linear(n_sub * d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d)
            )
        elif agg == "mean":
            pass
        else:
            raise ValueError(f"Unknown subbin agg '{agg}' (use mil|mlp|mean)")

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        # h: [M, N, d]   mask: [M, N] (bool, True = valid sub-bin)
        if self.agg == "mil":
            scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)  # [M, N]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)                                            # [M, N]
            feat = torch.einsum("mn,mnd->md", attn, h)                                      # [M, d]
            return feat, attn

        if self.agg == "mean":
            m = mask.unsqueeze(-1).float()
            feat = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            return feat, mask.float()

        # "mlp": zero masked sub-bins, then fixed-order concat -> MLP
        M, N, d = h.shape
        if N != self.n_sub:
            raise ValueError(f"agg='mlp' needs fixed N={self.n_sub}, got {N}")
        h = h * mask.unsqueeze(-1).float()
        feat = self.mlp(h.reshape(M, N * d))
        return feat, mask.float()


# ----------------------------------------------------------------------
# VisiumHD gene encoder
# ----------------------------------------------------------------------
@GENE_ENCODERS.register("subbin_gene_encoder")
class SubBinGeneEncoder(nn.Module):
    def __init__(self, gene_dim=2000, embed_dim=256, n_sub=4, d=512,
                 subbin_agg="mil", mil_hidden=128, per_bin_nonlinear=True,
                 gate_value=None,
                 config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        self.n_sub = n_sub
        self.d = d
        # gate_value=None -> learnable FidelityGate (default); a float -> fixed,
        # non-learnable gate (diagnostic, e.g. 0.5 = equal local/context).
        self._gate_value = gate_value

        # --- light, shared per-sub-bin encoder (kept intentionally shallow: a
        # single 16um bin is sparse/noisy; depth lives in the aggregator +
        # downstream LongNet, not here) ---
        self.input_norm = nn.LayerNorm(gene_dim)
        self.input_proj = nn.Linear(gene_dim, d)
        self.per_bin_nonlinear = per_bin_nonlinear
        if per_bin_nonlinear:
            self.bin_nonlin = nn.Linear(d, d)  # used as residual: h + GELU(lin(h))

        # --- learnable 2x2 (or NxN) slot positional embedding ---
        self.slot_emb = nn.Parameter(torch.zeros(n_sub, d))
        nn.init.normal_(self.slot_emb, std=0.02)

        # --- sub-bin aggregation head ---
        self.aggregator = SubBinAggregator(n_sub, d, agg=subbin_agg, mil_hidden=mil_hidden)

        # --- spatial-context fusion ---
        # default: shared learnable gate (identical to regular Visium);
        # if gate_value set: HD-only fixed-gate variant (does NOT touch shared code).
        if gate_value is None:
            self.context_processor = SubbinRelposEncoder(
                d, embed_dim, config_name=config_name, config_dict=config_dict, use_gate=True
            )
        else:
            self.context_processor = FixedGateContext(
                d, embed_dim, config_name=config_name, config_dict=config_dict,
                use_gate=True, gate_value=gate_value
            )

        # exposed for visualization: last target super-bin sub-bin attention [B, N]
        self.last_subbin_attn = None

    def _encode_subbins(self, sub: torch.Tensor):
        """sub: [M, N, gene_dim] -> (h [M, N, d], mask [M, N] bool)."""
        mask = (sub.abs().sum(dim=-1) > 0)                       # valid (non-empty) sub-bins
        h = self.input_proj(self.input_norm(sub))               # [M, N, d]
        if self.per_bin_nonlinear:
            h = h + F.gelu(self.bin_nonlin(h))
        h = h + self.slot_emb[: h.shape[1]].unsqueeze(0)        # + slot position
        # guard against an all-empty super-bin (avoid NaN in masked softmax)
        all_empty = (~mask).all(dim=1)
        if all_empty.any():
            mask = mask.clone()
            mask[all_empty, 0] = True
        return h, mask

    def _aggregate(self, sub: torch.Tensor):
        h, mask = self._encode_subbins(sub)
        return self.aggregator(h, mask)

    def forward(self, target_gene, context_genes, target_pos, context_pos):
        # Accept regular-Visium 2-D input too (treat as N=1).
        if target_gene.dim() == 2:
            target_gene = target_gene.unsqueeze(1)              # [B, 1, G]
        if context_genes.dim() == 3:
            context_genes = context_genes.unsqueeze(2)          # [B, C, 1, G]

        tgt_feat, tgt_attn = self._aggregate(target_gene)       # [B, d]
        self.last_subbin_attn = tgt_attn.detach()

        B, C, N, G = context_genes.shape
        ctx_feat, _ = self._aggregate(context_genes.reshape(B * C, N, G))
        ctx_feat = ctx_feat.reshape(B, C, -1)                   # [B, C, d]

        # shared gated spatial-context fusion -> (fused_gene, alpha_gene)
        return self.context_processor(tgt_feat, ctx_feat, target_pos, context_pos)


# ----------------------------------------------------------------------
# Aggregation variants (selectable via --gene-encoder-type, so no shared-entry
# edits are needed to switch the sub-bin aggregation head).
# ----------------------------------------------------------------------
@GENE_ENCODERS.register("subbin_gene_encoder_mlp")
class SubBinGeneEncoderMLP(SubBinGeneEncoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("subbin_agg", "mlp")
        super().__init__(*args, **kwargs)


@GENE_ENCODERS.register("subbin_gene_encoder_mean")
class SubBinGeneEncoderMean(SubBinGeneEncoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("subbin_agg", "mean")
        super().__init__(*args, **kwargs)


# ======================================================================
# Fixed-gate variants (HD-only diagnostic): force the context-fusion gate to a
# constant alpha (e.g. 0.5) instead of learning it -- to test whether the gate
# collapse (alpha~0.07) is what destroys local/biological structure.
# These live entirely here; the shared FidelityGate / MultiScaleAwareEncoder are
# NOT modified, so the regular-Visium pipeline is unaffected.
# ======================================================================
from src.models.image_encoders import GatedImageEncoder  # noqa: E402
from src.models.registry import IMAGE_ENCODERS            # noqa: E402


class FixedGate(nn.Module):
    """Non-learnable gate: fused = v*local + (1-v)*context, alpha == v (constant)."""
    def __init__(self, value=0.5):
        super().__init__()
        self.value = float(value)

    def forward(self, local_feat, context_vibe):
        fused = self.value * local_feat + (1.0 - self.value) * context_vibe
        alpha = torch.full((local_feat.size(0), 1), self.value,
                           device=local_feat.device, dtype=local_feat.dtype)
        return fused, alpha


class FixedGateContext(SubbinRelposEncoder):
    """SubbinRelposEncoder but with the learnable FidelityGate replaced by a
    constant gate. Reuses the parent forward (incl. relpos) unchanged."""
    def __init__(self, input_dim, embed_dim=256, gate_value=0.5, **kwargs):
        kwargs["use_gate"] = True
        super().__init__(input_dim, embed_dim, **kwargs)
        self.fidelity_gate = FixedGate(gate_value)   # swap in the constant gate
        if gate_value >= 1.0 and self.dilated_transformer is not None:
            # Pure-local: context contributes (1-alpha)=0, so the neighbor
            # transformer receives no gradient. Freeze it so DDP
            # (find_unused_parameters=False) and the optimizer don't choke on
            # zero-grad params. input_proj stays trainable (used by the local path).
            # (guard: neighbor_agg="vanilla" leaves dilated_transformer None.)
            for p in self.dilated_transformer.parameters():
                p.requires_grad = False


@IMAGE_ENCODERS.register("gated_image_encoder_fixedgate")
class GatedImageEncoderFixedGate(GatedImageEncoder):
    """Image encoder with a fixed (non-learnable) context gate. Inherits forward."""
    def __init__(self, backbone_type="vit_dino_v1", embed_dim=256,
                 config_name="LongNet_for_spatial", config_dict=None,
                 gate_value=0.5, **kwargs):
        nn.Module.__init__(self)
        self.image_encoder = IMAGE_ENCODERS.get(backbone_type)(embed_dim=embed_dim, **kwargs)
        self.context_processor = FixedGateContext(
            embed_dim, embed_dim, config_name=config_name, config_dict=config_dict,
            use_gate=True, gate_value=gate_value,
        )


@GENE_ENCODERS.register("subbin_gene_encoder_fixedgate")
class SubBinGeneEncoderFixedGate(SubBinGeneEncoder):
    """Sub-bin MIL gene encoder with the context gate fixed at 0.5."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("gate_value", 0.5)
        super().__init__(*args, **kwargs)


@GENE_ENCODERS.register("subbin_gene_encoder_local")
class SubBinGeneEncoderLocal(SubBinGeneEncoder):
    """PURE-LOCAL sub-bin MIL gene encoder: gate fixed at 1.0 (no spatial context;
    neighbor transformer frozen). Cuts the neighborhood-alignment shortcut."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("gate_value", 1.0)
        super().__init__(*args, **kwargs)


@IMAGE_ENCODERS.register("gated_image_encoder_local")
class GatedImageEncoderLocal(GatedImageEncoderFixedGate):
    """PURE-LOCAL image encoder: context gate fixed at 1.0 (neighbor transformer frozen)."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("gate_value", 1.0)
        super().__init__(*args, **kwargs)


# PURE-LOCAL + MIL at larger grids (48um N=9, 64um N=16): gate=1.0 + n_sub set.
@GENE_ENCODERS.register("subbin_gene_encoder_local_n9")
class SubBinGeneEncoderLocalN9(SubBinGeneEncoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("gate_value", 1.0); kwargs.setdefault("n_sub", 9)
        super().__init__(*args, **kwargs)


@GENE_ENCODERS.register("subbin_gene_encoder_local_n16")
class SubBinGeneEncoderLocalN16(SubBinGeneEncoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("gate_value", 1.0); kwargs.setdefault("n_sub", 16)
        super().__init__(*args, **kwargs)


# ----------------------------------------------------------------------
# Larger-grid variants (LEARNABLE gate, MIL): 3x3 (48um, N=9) / 4x4 (64um, N=16).
# Just fix n_sub so the slot positional embedding matches the data; gate stays
# learnable (default) so we can watch whether alpha recovers at bigger bin sizes.
# ----------------------------------------------------------------------
@GENE_ENCODERS.register("subbin_gene_encoder_n9")
class SubBinGeneEncoderN9(SubBinGeneEncoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("n_sub", 9)
        super().__init__(*args, **kwargs)


@GENE_ENCODERS.register("subbin_gene_encoder_n16")
class SubBinGeneEncoderN16(SubBinGeneEncoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("n_sub", 16)
        super().__init__(*args, **kwargs)


# ----------------------------------------------------------------------
# Ablation: NO MIL. Mean-pool the N sub-bins into ONE vector, then run the
# ORIGINAL regular-Visium gated_gene_encoder unchanged. Lets us compare the
# MIL sub-bin aggregation against the simple-average + default encoder, on the
# same sub-bin data. (Shared gene_encoders.FidelityGatedGeneEncoder is reused,
# not modified.)
# ----------------------------------------------------------------------
from src.models.gene_encoders import FidelityGatedGeneEncoder  # noqa: E402


@GENE_ENCODERS.register("avg_default_gene_encoder")
class AvgDefaultGeneEncoder(nn.Module):
    def __init__(self, gene_dim=2000, embed_dim=256,
                 config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        # This ablation deliberately uses the ORIGINAL, unmodified encoder; drop the
        # HD-only pos keys so the shared FidelityGatedGeneEncoder's EncoderConfig
        # (which receives config_dict) does not choke on unknown kwargs.
        if config_dict is not None:
            config_dict = {k: v for k, v in config_dict.items()
                           if k not in ("relative_pos", "pos_scale", "pos_weight", "neighbor_agg")}
        self.enc = FidelityGatedGeneEncoder(
            gene_dim=gene_dim, embed_dim=embed_dim,
            config_name=config_name, config_dict=config_dict,
        )

    @staticmethod
    def _pool(x):
        # x: [..., N, G] -> [..., G] masked mean over non-empty sub-bins
        m = (x.abs().sum(-1, keepdim=True) > 0).float()
        return (x * m).sum(-2) / m.sum(-2).clamp(min=1.0)

    def forward(self, target_gene, context_genes, target_pos, context_pos):
        if target_gene.dim() == 3:      # [B, N, G] -> [B, G]
            target_gene = self._pool(target_gene)
        if context_genes.dim() == 4:    # [B, C, N, G] -> [B, C, G]
            context_genes = self._pool(context_genes)
        return self.enc(target_gene, context_genes, target_pos, context_pos)

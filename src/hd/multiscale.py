# src/hd/multiscale.py
"""
VisiumHD-only neighbor encoder: relative-position (+ optional vanilla-attention).

This SUBCLASSES the shared, UNMODIFIED `MultiScaleAwareEncoder`. Nothing under
`src/models/` is touched, so the regular-Visium pipeline runs the original GitHub
code byte-for-byte.

Why a subclass is needed (and a plain config knob is not enough):
  * HD YAML configs carry extra keys: `relative_pos`, `pos_scale`, `pos_weight`,
    `neighbor_agg`. The pristine base forwards every config key straight into
    torchscale's `EncoderConfig(**args)`, which raises on those unknown kwargs.
  * So here we POP those keys out of `config_dict` BEFORE calling the base
    `__init__`, remember them on `self`, and override `forward` to use them.
  * Regular Visium never instantiates this class -- the HD gene encoder builds it
    only for sub-bin models; `gated_gene_encoder` keeps using the base directly.

Defaults (`relative_pos=False`, `pos_weight=0.3`, `neighbor_agg="attention"`)
reproduce the base `forward` exactly, so the subclass is behavior-neutral until
an HD config opts in.
"""
import torch
import torch.nn as nn

from src.models.multi_scale_blocks import MultiScaleAwareEncoder
from src.models.dilated_blocks import compute_2d_sincos_pos_embed


class SubbinRelposEncoder(MultiScaleAwareEncoder):
    """`MultiScaleAwareEncoder` + HD relative-position / vanilla-attention options."""

    def __init__(self, input_dim, embed_dim=256, use_gate=True,
                 config_name="LongNet_for_spatial", config_dict=None,
                 relative_pos=False, pos_scale=None, pos_weight=0.3):
        # 1) Pull the HD-only knobs out of config_dict so the pristine base never
        #    hands them to EncoderConfig (which would raise on unknown kwargs).
        #    Explicit kwargs act as fallbacks when a key is absent from the YAML.
        neighbor_agg = "attention"
        cleaned = None
        if config_dict is not None:
            cleaned = dict(config_dict)  # copy: never mutate the caller's dict
            relative_pos = cleaned.pop("relative_pos", relative_pos)
            pos_scale = cleaned.pop("pos_scale", pos_scale)
            pos_weight = cleaned.pop("pos_weight", pos_weight)
            neighbor_agg = cleaned.pop("neighbor_agg", neighbor_agg)

        # 2) Let the UNMODIFIED base build input_proj, the dilated transformer
        #    (from the cleaned dict) and the fidelity gate.
        super().__init__(input_dim, embed_dim, use_gate=use_gate,
                         config_name=config_name, config_dict=cleaned)

        # 3) Remember the HD knobs for the overridden forward.
        self.relative_pos = bool(relative_pos)
        self.pos_scale = pos_scale
        self.pos_weight = pos_weight
        self.neighbor_agg = neighbor_agg

        # 4) Optional textbook full-attention ablation (only the vanilla YAML sets
        #    neighbor_agg="vanilla"): swap the dilated LongNet for a plain encoder,
        #    matched on depth/width/heads/ffn.
        self.vanilla_transformer = None
        if self.neighbor_agg == "vanilla":
            cfg = cleaned or {}
            n_layers = int(cfg.get("encoder_layers", 4))
            n_heads = int(cfg.get("encoder_attention_heads", 8))
            ffn = int(cfg.get("encoder_ffn_embed_dim", 4 * embed_dim))
            drop = float(cfg.get("dropout", 0.1))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn,
                dropout=drop, activation="gelu", batch_first=True, norm_first=True,
            )
            self.vanilla_transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.dilated_transformer = None  # drop the base-built dilated path

    def forward(self, target_feat, context_feats, target_pos, context_pos):
        # Mirrors the base forward; the ONLY differences are the relative-position
        # branch (Step 2) and the optional vanilla-attention path -- both HD opt-in.
        batch_size = target_feat.shape[0]
        num_context = context_feats.shape[1]

        # --- Step 1: identity path (pure center features, kept separate) ---
        target_local = self.input_proj(target_feat)  # [B, E]
        context_flat = context_feats.reshape(-1, self.input_dim)
        context_proj = self.input_proj(context_flat).reshape(batch_size, num_context, self.embed_dim)  # [B, C, E]

        # --- Step 2: neighbor environment with (optional) relative positions ---
        if self.relative_pos:
            # Encode neighbors RELATIVE to the center, so the embedding carries
            # local geometry only (no absolute "where in the slide" shortcut).
            pos_input = context_pos - target_pos.unsqueeze(1)  # [B, C, 2]
            if self.pos_scale:
                pos_input = pos_input / float(self.pos_scale)
        else:
            # Base behavior: encode absolute (global-normalized) coords.
            pos_input = context_pos
        context_pos_emb = compute_2d_sincos_pos_embed(pos_input, self.embed_dim, device=context_proj.device)
        context_input = context_proj + self.pos_weight * context_pos_emb

        # Transformer sees neighbors only (target excluded) -> "pure background vibe".
        if self.neighbor_agg == "vanilla":
            neighbor_encoded = self.vanilla_transformer(context_input)  # [B, C, E]
        else:
            out_dict = self.dilated_transformer(src_tokens=None, token_embeddings=context_input)
            neighbor_encoded = out_dict["encoder_out"]  # [B, C, E]

        neighbor_vibe = torch.mean(neighbor_encoded, dim=1)  # [B, E]

        # --- Step 3: adaptive gated fusion (identical to base) ---
        if self.use_gate:
            output, alpha = self.fidelity_gate(target_local, neighbor_vibe)
        else:
            combined = torch.cat([target_local, neighbor_vibe], dim=-1)
            output = self.fusion_mlp(combined) + target_local
            alpha = None
        return output, alpha

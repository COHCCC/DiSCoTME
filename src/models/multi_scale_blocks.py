# src/models/multi_scale_blocks.py

import torch
import torch.nn as nn
from .dilated_blocks import build_dilated_transformer, compute_2d_sincos_pos_embed, DilatedConfigs
from torchscale.architecture.config import EncoderConfig
from .dilated_blocks import DilatedEncoder

# ----------------------------------------------------------------------
# 1. Fidelity Gate: Adaptive Gating Fusion Module
# ----------------------------------------------------------------------
class FidelityGate(nn.Module):
    """
    Adaptive Gate: Determines 'fidelity' based on local features.
    In vascular regions, it should learn high Alpha (preserving local details);
    In uniform tissue regions, it should learn low Alpha (absorbing the environment).
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, local_feat, context_vibe):
        # alpha -> 1: Preserve local identity (safeguard morphological details)
        # alpha -> 0: Merge with environment (differentiate T-cell niches)
        alpha = self.gate(local_feat) # [B, 1]
        fused = alpha * local_feat + (1 - alpha) * context_vibe
        return fused, alpha


# ----------------------------------------------------------------------
# 2. MultiScaleAwareEncoder: Improved Multi-Scale Spatial Encoder
# ----------------------------------------------------------------------
class MultiScaleAwareEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256, use_gate=True, 
                 config_name="LongNet_for_spatial", config_dict=None):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_gate = use_gate

        # Projection Layer
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # 1. Environment Modeling Transformer (models neighbors only)
        if config_dict is not None:
            self.dilated_transformer = self._build_from_dict(config_dict)
        else:
            self.dilated_transformer = build_dilated_transformer(config_name)

        # 2. Fusion Logic (Gate)
        if self.use_gate:
            self.fidelity_gate = FidelityGate(embed_dim)
        else:
            # Alternative Concat + MLP logic
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

    def _build_from_dict(self, config_dict):
        """Directly build DilatedEncoder from a configuration dictionary"""
        args = config_dict.copy()
        args.setdefault('dropout', 0.1)
        args.setdefault('drop_path_rate', 0.1)
        arch = EncoderConfig(**args)
        return DilatedEncoder(arch)

    def forward(self, target_feat, context_feats, target_pos, context_pos):
        """
        Args:
            target_feat: [B, input_dim] - Original features of the center spot
            context_feats: [B, num_context, input_dim] - 16-18 sampled neighbors
            target_pos: [B, 2] - Center coordinates
            context_pos: [B, num_context, 2] - Neighbor coordinates
        """
        batch_size = target_feat.shape[0]
        num_context = context_feats.shape[1]

        # --- Step 1: Projection and Identity Extraction (Identity Path) ---
        # target_local represents pure morphological or genetic features, 
        # kept separate from neighbor averaging to preserve detail.
        target_local = self.input_proj(target_feat) # [B, E]

        # Project neighbor features
        context_flat = context_feats.reshape(-1, self.input_dim)
        context_proj = self.input_proj(context_flat)
        context_proj = context_proj.reshape(batch_size, num_context, self.embed_dim) # [B, C, E]

        # --- Step 2: Neighbor Environment Modeling (Niche/Environment Path) ---
        # Positional Encoding: Allows neighbors to perceive relative positions
        context_pos_emb = compute_2d_sincos_pos_embed(context_pos, self.embed_dim, device=context_proj.device)
        context_input = context_proj + 0.3 * context_pos_emb

        # [Key Design]: Transformer input only contains neighbors, excluding the Target.
        # This ensures the environmental vector represents a "pure background vibe" 
        # without diluting the specific details of the Target itself.
        out_dict = self.dilated_transformer(src_tokens=None, token_embeddings=context_input)
        neighbor_encoded = out_dict["encoder_out"] # [B, C, E]

        # Summarize environmental information (Neighborhood Vibe)
        # Mean pooling extracts commonality across the neighbors
        neighbor_vibe = torch.mean(neighbor_encoded, dim=1) # [B, E]

        # --- Step 3: Adaptive Gated Fusion (Fusion Path) ---
        if self.use_gate:
            # Compare "pure local identity" with "environmental background" to automatically learn Alpha
            output, alpha = self.fidelity_gate(target_local, neighbor_vibe)
        else:
            # Fallback Concat logic
            combined = torch.cat([target_local, neighbor_vibe], dim=-1)
            output = self.fusion_mlp(combined) + target_local
            alpha = None

        # Return fused features and fidelity weight (alpha)
        return output, alpha
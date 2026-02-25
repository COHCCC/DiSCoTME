# src/models/gene_encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import GENE_ENCODERS
from .multi_scale_blocks import MultiScaleAwareEncoder

# ==========================================
# 1. [[Model default]  Gated Gene Encoder with MLP + Residuals
# ==========================================
@GENE_ENCODERS.register("gated_gene_encoder")
class FidelityGatedGeneEncoder(nn.Module):
    """
    Adaptive Gated Gene Encoder:
    - Uses MLP + Residuals to extract cellular identity.
    - Uses LongNet to extract spatial ecosystem (Niche) information.
    - Returns fused features and alpha_gene fidelity weights.
    """
    def __init__(self, gene_dim=2000, embed_dim=256, config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(gene_dim)
        self.input_proj = nn.Linear(gene_dim, 512)
        
        # Optimized gene local encoder
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        
        # Spatial context processor (shares logic with image-side)
        self.context_processor = MultiScaleAwareEncoder(
            512, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict,
            use_gate=True
        )
        
    def forward(self, target_gene, context_genes, target_pos, context_pos):
        # --- 1. Normalization ---
        target_gene = self.input_norm(target_gene)
        B, C, D = context_genes.shape
        context_genes_flat = self.input_norm(context_genes.reshape(B*C, D))
        
        # --- 2. Extract Local Identity (MLP + Residuals) ---
        target_proj = self.input_proj(target_gene) 
        target_feat = target_proj + self.mlp(target_proj)

        context_proj = self.input_proj(context_genes_flat)
        context_feat = context_proj + self.mlp(context_proj)
        context_feat = context_feat.reshape(B, C, -1)

        # --- 3. Fidelity-Gated Fusion ---
        # Returns: Fused gene features, alpha_gene fidelity weight
        fused_gene, alpha_gene = self.context_processor(target_feat, context_feat, target_pos, context_pos)
        
        return fused_gene, alpha_gene
# ==========================================
# 2.  MLP with Residuals
# ==========================================
@GENE_ENCODERS.register("gene_mlp_residual")
class GeneEncoderMLPResidual(nn.Module):
    """
    Standard ContextAwareGeneEncoder:
    - Input Norm -> Proj
    - Residual Connection: feat = proj + mlp(proj)
    """
    def __init__(self, gene_dim=2000, embed_dim=256, config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(gene_dim)
        self.input_proj = nn.Linear(gene_dim, 512)
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        self.context_processor = MultiScaleAwareEncoder(
            512, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict
        )
        
    def forward(self, target_gene, context_genes, target_pos, context_pos):
        target_gene = self.input_norm(target_gene)
        B, C, D = context_genes.shape
        context_genes = self.input_norm(context_genes.reshape(B*C, D)).reshape(B, C, D)
        
        target_proj = self.input_proj(target_gene) 
        target_feat = target_proj + self.mlp(target_proj)  # Residual Connection

        context_proj = self.input_proj(context_genes.reshape(B*C, D))
        context_feat = context_proj + self.mlp(context_proj)  # Residual Connection
        context_feat = context_feat.reshape(B, C, -1)

        return self.context_processor(target_feat, context_feat, target_pos, context_pos)  


# ==========================================
# 3. [VARIANT] MLP with Weighted Residual (0.1)
# ==========================================
@GENE_ENCODERS.register("gene_mlp_weighted_residual")
class GeneEncoderWeightedResidual(nn.Module):
    """
    Weighted residual version (0.1) to prevent learning bias.
    """
    def __init__(self, gene_dim=2000, embed_dim=256, config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        self.input_norm = nn.LayerNorm(gene_dim)
        self.input_proj = nn.Linear(gene_dim, 512)
        self.mlp = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(512, 512), nn.LayerNorm(512),
        )
        self.context_processor = MultiScaleAwareEncoder(
            512, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict
        )

    def forward(self, target_gene, context_genes, target_pos, context_pos):
        target_gene = self.input_norm(target_gene)
        B, C, D = context_genes.shape
        context_genes = self.input_norm(context_genes.reshape(B*C, D)).reshape(B, C, D)
        
        target_proj = self.input_proj(target_gene)
        target_feat = target_proj + 0.1 * self.mlp(target_proj)  # 0.1 Weight
        
        context_proj = self.input_proj(context_genes.reshape(B*C, D))
        context_feat = context_proj + 0.1 * self.mlp(context_proj)  # 0.1 Weight
        context_feat = context_feat.reshape(B, C, -1)
        
        return self.context_processor(target_feat, context_feat, target_pos, context_pos)


# ==========================================
# 4. [VARIANT] Transformer Encoder (Intra-spot)
# ==========================================
@GENE_ENCODERS.register("gene_transformer_intra")
class GeneEncoderTransformerIntra(nn.Module):
    """
    Direct TransformerEncoder on 512-dim features.
    """
    def __init__(self, gene_dim=512, embed_dim=256, num_heads=8, config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        assert gene_dim % num_heads == 0, "gene_dim must be divisible by num_heads"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gene_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=2048, dropout=0.1, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.context_processor = MultiScaleAwareEncoder(
            gene_dim, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict
        )

    def forward(self, target_gene, context_genes, target_pos, context_pos):
        target_gene = target_gene.unsqueeze(1)
        combined = torch.cat([target_gene, context_genes], dim=1)
        
        out = self.transformer(combined)  # Transformer encode
        
        target_out = out[:, 0, :]
        context_out = out[:, 1:, :]
        return self.context_processor(target_out, context_out, target_pos, context_pos)


# ==========================================
# 5. [VARIANT] Gene Identity Transformer
# ==========================================
@GENE_ENCODERS.register("gene_identity_transformer")
class GeneEncoderIdentityTransformer(nn.Module):
    """
    Transformer version with gene identity and weighted pooling.
    """
    def __init__(self, gene_dim=200, embed_dim=256, num_heads=4, num_layers=2, 
                 config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        gene_embed_dim = 64
        self.gene_dim = gene_dim
        self.gene_embed = nn.Linear(1, gene_embed_dim)
        self.gene_id = nn.Parameter(torch.randn(1, gene_dim, gene_embed_dim))
        self.gene_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=gene_embed_dim, nhead=num_heads, 
                dim_feedforward=gene_embed_dim*4, batch_first=True, dropout=0.1
            ),
            num_layers=num_layers
        )
        self.proj = nn.Linear(gene_embed_dim, embed_dim)
        self.context_processor = MultiScaleAwareEncoder(
            embed_dim, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict
        )
   
    def encode_genes(self, genes):
        weights = F.softmax(genes, dim=-1)
        x = genes.unsqueeze(-1)
        x = self.gene_embed(x)
        x = x + self.gene_id
        x = self.gene_transformer(x)
        weights = weights.unsqueeze(-1)
        x = (x * weights).sum(dim=1)
        return self.proj(x)
    
    def forward(self, target_gene, context_genes, target_pos, context_pos):
        target_feat = self.encode_genes(target_gene)
        B, C, G = context_genes.shape
        context_flat = context_genes.reshape(B * C, G)
        context_feats = self.encode_genes(context_flat).reshape(B, C, -1)
        return self.context_processor(target_feat, context_feats, target_pos, context_pos)


# ==========================================
# 6. [VARIANT] Modular MLP
# ==========================================
@GENE_ENCODERS.register("gene_mlp_modular")
class GeneEncoderModular(nn.Module):
    """
    Modular implementation with internal GeneEncoder and ContextAware wrapper.
    """
    class _InnerGeneEncoder(nn.Module):
        def __init__(self, input_dim=512, hidden_dim=1024, output_dim=256, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.ln2 = nn.LayerNorm(output_dim)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.shortcut = nn.Linear(input_dim, output_dim)
            
        def forward(self, x):
            identity = self.shortcut(x)
            out = self.fc1(x)
            out = self.ln1(out)
            out = self.gelu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.ln2(out)
            return self.gelu(out + identity)

    def __init__(self, gene_dim=512, embed_dim=256, config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        self.gene_mlp = self._InnerGeneEncoder(input_dim=gene_dim, output_dim=embed_dim)
        self.context_processor = MultiScaleAwareEncoder(
            embed_dim, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict
        )

    def forward(self, target_gene, context_genes, target_pos, context_pos):
        batch_size = target_gene.shape[0]
        num_context = context_genes.shape[1]
        gene_dim = target_gene.shape[1]
        
        target_feat = self.gene_mlp(target_gene)
        context_flat = context_genes.reshape(-1, gene_dim)
        context_feats_flat = self.gene_mlp(context_flat)
        context_feats = context_feats_flat.reshape(batch_size, num_context, -1)
        
        return self.context_processor(target_feat, context_feats, target_pos, context_pos)


# ==========================================
# 7. [VARIANT] Simple Linear Projection
# ==========================================
@GENE_ENCODERS.register("gene_simple_linear")
class GeneEncoderSimple(nn.Module):
    """
    Baseline implementation with basic linear projection.
    """
    def __init__(self, gene_dim=512, embed_dim=256, config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(gene_dim, embed_dim)
        self.context_processor = MultiScaleAwareEncoder(
            gene_dim, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict
        )

    def forward(self, target_gene, context_genes, target_pos, context_pos):
        # Basic projection without MLP or residuals
        return self.context_processor(target_gene, context_genes, target_pos, context_pos)
    

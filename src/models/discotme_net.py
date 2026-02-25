# src/models/discotme_net.py

import torch
import torch.nn as nn
from .registry import Registry, IMAGE_ENCODERS, GENE_ENCODERS
from . import image_encoders
from . import gene_encoders

# ==============================================================================
# Model Registry
# ==============================================================================
MODELS = Registry("models")

# ==============================================================================
# 1. Standard Version [Model default]
# ==============================================================================
@MODELS.register("standard_discotme")
class MultiScaleMultiModalModel(nn.Module):
    def __init__(self, img_enc_name, gene_enc_name, proj_dim=128, **kwargs):
        super().__init__()
        
        # Extract parameters from kwargs
        img_args = kwargs.get('img_args', {})
        gene_args = kwargs.get('gene_args', {})

        # Initialize Image Encoder (Standard or Confidence-based)
        self.img_encoder = IMAGE_ENCODERS.get(img_enc_name)(**img_args)
        
        # Initialize Gene Encoder
        self.gene_encoder = GENE_ENCODERS.get(gene_enc_name)(**gene_args)

        self.img_proj = nn.Sequential(
            nn.Linear(256, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(256, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, batch):
        target_img = batch['target_img']
        target_gene = batch['target_gene']
        target_pos = batch['target_pos']
        context_imgs = batch['context_imgs']
        context_genes = batch['context_genes']
        context_pos = batch['context_pos']

        # 1. Image Encoder
        img_out = self.img_encoder(target_img, context_imgs, target_pos, context_pos)

        alpha_img = None
        if isinstance(img_out, tuple):
            img_feat = img_out[0]
            if len(img_out) == 2:
                # Gated Version: (feat, alpha)
                alpha_img = img_out[1]
            elif len(img_out) >= 4:
                # Legacy Confidence Version: (feat, conf, raw, teacher)
                alpha_img = img_out[1]
                img_raw = img_out[2]
                img_teacher = img_out[3]
        else:
            img_feat = img_out
        
        # 2. Gene Encoder
        gene_out = self.gene_encoder(target_gene, context_genes, target_pos, context_pos)
        
        alpha_gene = None
        if isinstance(gene_out, tuple):
            gene_feat = gene_out[0]
            alpha_gene = gene_out[1] if len(gene_out) > 1 else None
        else:
            gene_feat = gene_out

        # 3. Projection
        img_emb = self.img_proj(img_feat)
        gene_emb = self.gene_proj(gene_feat)

        # 4. Return (Uniformly returns 4 values, alpha can be None)
        return img_emb, gene_emb, alpha_img, alpha_gene
    
# ==============================================================================
# 2. FactorCL Version (Factorized DiSCoTME)
# ==============================================================================
@MODELS.register("factorized_discotme")
class FactorizedMultiModalModel(MultiScaleMultiModalModel):
    def __init__(self, img_enc_name, gene_enc_name, proj_dim=128, **kwargs):
        super().__init__(img_enc_name, gene_enc_name, proj_dim, **kwargs)
        
        # Remove parent class single head
        del self.img_proj 
        
        # 1. Projectors (Encoder Head)
        self.img_proj_shared = nn.Linear(256, proj_dim)
        self.img_proj_unique = nn.Linear(256, proj_dim)
        
        # 2. Decoders (Reconstruction Head)
        # Note: 256 is the original dimension output by DINO
        self.decoder_shared = nn.Linear(proj_dim, 256) 
        self.decoder_unique = nn.Linear(proj_dim, 256)

    def forward(self, batch):
        target_img = batch['target_img']
        target_gene = batch['target_gene']
        target_pos = batch['target_pos']
        context_imgs = batch['context_imgs']
        context_genes = batch['context_genes']
        context_pos = batch['context_pos']

        # --- 1. Encoder (Reusing logic) ---
        img_out = self.img_encoder(target_img, context_imgs, target_pos, context_pos)
        gene_out = self.gene_encoder(target_gene, context_genes, target_pos, context_pos)

        # Unpack Image
        alpha_img = None
        if isinstance(img_out, tuple):
            img_feat = img_out[0]
            alpha_img = img_out[1] if len(img_out) >= 2 else None
        else:
            img_feat = img_out
        
        # Unpack Gene
        alpha_gene = None
        if isinstance(gene_out, tuple):
            gene_feat = gene_out[0]
            alpha_gene = gene_out[1] if len(gene_out) >= 2 else None
        else:
            gene_feat = gene_out

        # --- 2. Factorized Projection (Core implementation) ---
        
        z_img_shared = self.img_proj_shared(img_feat)
        z_img_unique = self.img_proj_unique(img_feat)
        z_gene_shared = self.gene_proj(gene_feat)
        
        # Reconstruction Flow (For loss calculation)
        rec_shared = self.decoder_shared(z_img_shared)
        rec_unique = self.decoder_unique(z_img_unique)
        
        # Return 8 values
        return z_img_shared, z_img_unique, z_gene_shared, alpha_img, alpha_gene, rec_shared, rec_unique, img_feat
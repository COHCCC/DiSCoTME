import torch
import torch.nn as nn
from .registry import Registry, IMAGE_ENCODERS, GENE_ENCODERS
from . import image_encoders
from . import gene_encoders

# ==============================================================================
# 新增：模型注册表
# ==============================================================================
MODELS = Registry("models")

# ==============================================================================
# 1. 标准版 (Standard DiSCoTME) - 你原有的代码
# ==============================================================================
@MODELS.register("standard_discotme")
class MultiScaleMultiModalModel(nn.Module):
    def __init__(self, img_enc_name, gene_enc_name, proj_dim=128, **kwargs):
        super().__init__()
        
        # 从 kwargs 提取参数
        img_args = kwargs.get('img_args', {})
        gene_args = kwargs.get('gene_args', {})

        # 初始化 Image Encoder (可能是普通的，也可能是带置信度的)
        self.img_encoder = IMAGE_ENCODERS.get(img_enc_name)(**img_args)
        
        # 初始化 Gene Encoder
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
# # ==============================================================================
# # 换成clip风格的porjector
# # ==============================================================================
        
#         self.img_proj = nn.Linear(256, proj_dim)
#         self.gene_proj = nn.Linear(256, proj_dim)

    def forward(self, batch):
        target_img = batch['target_img']
        target_gene = batch['target_gene']
        target_pos = batch['target_pos']
        context_imgs = batch['context_imgs']
        context_genes = batch['context_genes']
        context_pos = batch['context_pos']

        img_out = self.img_encoder(target_img, context_imgs, target_pos, context_pos)

        alpha_img = None
        if isinstance(img_out, tuple):
            img_feat = img_out[0]
            if len(img_out) == 2:
                # Gated 版本：(feat, alpha)
                alpha_img = img_out[1]
            elif len(img_out) >= 4:
                # 旧的 Confidence 版本：(feat, conf, raw, teacher)
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

        # 3. 投影
        img_emb = self.img_proj(img_feat)
        gene_emb = self.gene_proj(gene_feat)

        # 4. 返回（统一返回 4 个值，alpha 可能是 None）
        return img_emb, gene_emb, alpha_img, alpha_gene
    
# ==============================================================================
# 2. FactorCL 版 (Factorized DiSCoTME) - 新增代码
# ==============================================================================
@MODELS.register("factorized_discotme")
class FactorizedMultiModalModel(MultiScaleMultiModalModel):
    def __init__(self, img_enc_name, gene_enc_name, proj_dim=128, **kwargs):
        super().__init__(img_enc_name, gene_enc_name, proj_dim, **kwargs)
        
        del self.img_proj # 删除父类单头
        
        # 1. Projectors (Encoder Head)
        self.img_proj_shared = nn.Linear(256, proj_dim)
        self.img_proj_unique = nn.Linear(256, proj_dim)
        
        # 2. Decoders (Reconstruction Head)
        # 既然是 Linear Projection，逆变换最好也是 Linear
        # 256 是 DINO 输出的原始维度
        self.decoder_shared = nn.Linear(proj_dim, 256) 
        self.decoder_unique = nn.Linear(proj_dim, 256)

    def forward(self, batch):
        target_img = batch['target_img']
        target_gene = batch['target_gene']
        target_pos = batch['target_pos']
        context_imgs = batch['context_imgs']
        context_genes = batch['context_genes']
        context_pos = batch['context_pos']

        # --- 1. Encoder (复用逻辑) ---
        img_out = self.img_encoder(target_img, context_imgs, target_pos, context_pos)
        gene_out = self.gene_encoder(target_gene, context_genes, target_pos, context_pos)

        # 解包 Image
        alpha_img = None
        if isinstance(img_out, tuple):
            img_feat = img_out[0]
            alpha_img = img_out[1] if len(img_out) >= 2 else None
        else:
            img_feat = img_out
        
        # 解包 Gene
        alpha_gene = None
        if isinstance(gene_out, tuple):
            gene_feat = gene_out[0]
            alpha_gene = gene_out[1] if len(gene_out) >= 2 else None
        else:
            gene_feat = gene_out

        # --- 2. Factorized Projection (核心差异) ---
        
        z_img_shared = self.img_proj_shared(img_feat)
        z_img_unique = self.img_proj_unique(img_feat)
        z_gene_shared = self.gene_proj(gene_feat)
        
        # Reconstruction Flow (为了计算 Loss)
        # 我们直接在 Forward 里算好 recon 传出去，或者只传 z 出去在 Trainer 算
        # 为了代码整洁，我们在 Trainer 里调用 decoder 比较好，或者在这里算好：
        
        rec_shared = self.decoder_shared(z_img_shared)
        rec_unique = self.decoder_unique(z_img_unique)
        
        # 返回 7 个值
        return z_img_shared, z_img_unique, z_gene_shared, alpha_img, alpha_gene, rec_shared, rec_unique, img_feat
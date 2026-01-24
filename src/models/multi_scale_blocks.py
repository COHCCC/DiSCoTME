# src/models/multi_scale_blocks.py

import torch
import torch.nn as nn
from .dilated_blocks import build_dilated_transformer, compute_2d_sincos_pos_embed, DilatedConfigs
from torchscale.architecture.config import EncoderConfig
from .dilated_blocks import DilatedEncoder

# ----------------------------------------------------------------------
# 1. Fidelity Gate: 自适应门控融合模块
# ----------------------------------------------------------------------
class FidelityGate(nn.Module):
    """
    自适应门控：根据局部特征决定“保真度”。
    在血管区域，它应学会高 Alpha（保留局部）；
    在均一组织区，它应学会低 Alpha（吸收环境）。
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
        # alpha -> 1: 保真局部 (守护形态细节)
        # alpha -> 0: 融入环境 (区分 T 细胞生态位)
        alpha = self.gate(local_feat) # [B, 1]
        fused = alpha * local_feat + (1 - alpha) * context_vibe
        return fused, alpha


# class FidelityGate(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         # [修改点 1] 输入维度变为 2 倍，因为我们要拼接 local 和 context
#         self.gate = nn.Sequential(
#             nn.Linear(embed_dim * 2, 128),  # 这里的输入变成了 embed_dim * 2
#             nn.GELU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, local_feat, context_vibe):
#         # [修改点 2] 拼接！让 Gate 看到对比
#         # local_feat: [B, E]
#         # context_vibe: [B, E]
#         decision_input = torch.cat([local_feat, context_vibe], dim=-1) # [B, 2*E]
        
#         alpha = self.gate(decision_input) # [B, 1]
        
#         # 融合公式不变
#         fused = alpha * local_feat + (1 - alpha) * context_vibe
#         return fused, alpha

# ----------------------------------------------------------------------
# 2. MultiScaleAwareEncoder: 改进后的多尺度空间编码器
# ----------------------------------------------------------------------
class MultiScaleAwareEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256, use_gate=True, 
                 config_name="LongNet_for_spatial", config_dict=None):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_gate = use_gate

        # 投影层
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # 1. 环境建模 Transformer (只对邻居建模)
        if config_dict is not None:
            self.dilated_transformer = self._build_from_dict(config_dict)
        else:
            self.dilated_transformer = build_dilated_transformer(config_name)

        # 2. 融合逻辑 (Gate)
        if self.use_gate:
            self.fidelity_gate = FidelityGate(embed_dim)
        else:
            # 备选的 Concat + MLP 逻辑
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

    def _build_from_dict(self, config_dict):
        """从字典直接构建 DilatedEncoder"""
        args = config_dict.copy()
        args.setdefault('dropout', 0.1)
        args.setdefault('drop_path_rate', 0.1)
        arch = EncoderConfig(**args)
        return DilatedEncoder(arch)

    def forward(self, target_feat, context_feats, target_pos, context_pos):
        """
        Args:
            target_feat: [B, input_dim] - 中心 Spot 的原始特征
            context_feats: [B, num_context, input_dim] - 16-18个采样邻居
            target_pos: [B, 2] - 中心坐标
            context_pos: [B, num_context, 2] - 邻居坐标
        """
        batch_size = target_feat.shape[0]
        num_context = context_feats.shape[1]

        # --- Step 1: 投影与身份提取 (Identity Path) ---
        # target_local 是纯净的形态学或基因学特征，不参与邻居的平均化
        target_local = self.input_proj(target_feat) # [B, E]

        # 投影邻居特征
        context_flat = context_feats.reshape(-1, self.input_dim)
        context_proj = self.input_proj(context_flat)
        context_proj = context_proj.reshape(batch_size, num_context, self.embed_dim) # [B, C, E]

        # --- Step 2: 邻居环境建模 (Niche/Environment Path) ---
        # 位置编码：让邻居感知彼此的相对位置
        context_pos_emb = compute_2d_sincos_pos_embed(context_pos, self.embed_dim, device=context_proj.device)
        context_input = context_proj + 0.3 * context_pos_emb

        # [关键设计]：Transformer 输入只有邻居，不包含 Target！
        # 这样输出的环境向量代表的是“纯粹的背景氛围”，不会稀释 Target 自身的细节。
        out_dict = self.dilated_transformer(src_tokens=None, token_embeddings=context_input)
        neighbor_encoded = out_dict["encoder_out"] # [B, C, E]

        # 汇总环境信息 (Neighborhood Vibe)
        # 这里使用 Mean Pooling 提取 16 个邻居的共性
        neighbor_vibe = torch.mean(neighbor_encoded, dim=1) # [B, E]

        # --- Step 3: 自适应门控融合 (Fusion Path) ---
        if self.use_gate:
            # 用“纯净局部身份”去对比“环境背景”，自动学习 Alpha
            output, alpha = self.fidelity_gate(target_local, neighbor_vibe)
        else:
            # 兼容 Concat 逻辑
            combined = torch.cat([target_local, neighbor_vibe], dim=-1)
            output = self.fusion_mlp(combined) + target_local
            alpha = None

        # 返回融合特征和保真度值
        return output, alpha
    
# class MultiScaleAwareEncoder(nn.Module):
#     """
#     支持两种配置方式:
#     1. config_name: 字符串，从 DilatedConfigs 字典中查找
#     2. config_dict: 字典，直接使用（优先级更高）
#     """
#     def __init__(self, input_dim, embed_dim=256, config_name="LongNet_for_spatial", config_dict=None):
#         super().__init__()
#         self.input_dim = input_dim
#         self.embed_dim = embed_dim

#         self.input_proj = nn.Linear(input_dim, embed_dim)
        
#         # 构建 Dilated Transformer
#         if config_dict is not None:
#             # 直接使用传入的字典配置
#             self.dilated_transformer = self._build_from_dict(config_dict)
#         else:
#             # 使用预设名称
#             self.dilated_transformer = build_dilated_transformer(config_name)

#     def _build_from_dict(self, config_dict):
#         """从字典直接构建 DilatedEncoder"""
#         # 确保必要的默认值
#         args = config_dict.copy()
#         args.setdefault('dropout', 0.1)
#         args.setdefault('drop_path_rate', 0.1)
        
#         arch = EncoderConfig(**args)
#         model = DilatedEncoder(arch)
        
#         n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"Create DilatedTransformer from config_dict, # params= {n_params/1e6:.2f}M")
        
#         return model

#     def forward(self, target_feat, context_feats, target_pos, context_pos):
#         """
#         Using Dilated Attention to capture multi-scale context
#         """
#         batch_size = target_feat.shape[0]
#         num_context = context_feats.shape[1]

#         target_proj = self.input_proj(target_feat)  # [B, E]

#         context_flat = context_feats.reshape(-1, self.input_dim)
#         context_proj = self.input_proj(context_flat)
#         context_proj = context_proj.reshape(batch_size, num_context, self.embed_dim)

#         # Positional Embeddings
#         target_pos_emb = compute_2d_sincos_pos_embed(target_pos, self.embed_dim, device=target_proj.device)
#         target_proj = target_proj + 0.3 * target_pos_emb

#         context_pos_emb = compute_2d_sincos_pos_embed(context_pos, self.embed_dim, device=context_proj.device)
#         context_proj = context_proj + 0.3 * context_pos_emb

#         target_proj = target_proj.unsqueeze(1)                           # [B, 1, E]
#         combined = torch.cat([target_proj, context_proj], dim=1)         # [B, 1+C, E]

#         out_dict = self.dilated_transformer(src_tokens=None, token_embeddings=combined)
#         seq_output = out_dict["encoder_out"]                             # [B, 1+C, E]

#         target_output = seq_output[:, 0]                                 # [B, E]
#         return target_output
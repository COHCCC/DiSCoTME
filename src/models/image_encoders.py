# src/models/image_encoders.py

import torch
import torch.nn as nn
import timm
from .registry import IMAGE_ENCODERS
from .multi_scale_blocks import MultiScaleAwareEncoder

# ==========================================
# 1. [01/18 BEST PREFORM] 当前使用的版本
# ==========================================
@IMAGE_ENCODERS.register("gigapath_frozen_v1")
class ImageEncoderGigaPath(nn.Module):
    """
    对应你代码中未注释的 ImageEncoder:
    - Backbone: GigaPath
    - Freeze all except last 3 blocks + norm
    - Proj: Linear -> LayerNorm -> GELU -> Linear
    """
    def __init__(self, model_name="hf_hub:prov-gigapath/prov-gigapath", pretrained=True, embed_dim=256, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.backbone.set_grad_checkpointing(enable=True)
        
        # 先全部冻结
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 只解冻最后3层 + norm
        unfreeze_indices = [f"blocks.{i}." for i in range(37, 40)]
        unfreeze_indices.append("norm.")
        
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in unfreeze_indices):
                param.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.proj(feat)


# ==========================================
# 2. 带置信度的 Gigapath Encoder 
# ==========================================
@IMAGE_ENCODERS.register("gigapath_with_confidence")
class ImageEncoderWithConfidence(nn.Module):
    """
    带 confidence_head 的版本，用于 Distillation
    """
    def __init__(self, model_name="hf_hub:prov-gigapath/prov-gigapath", pretrained=True, embed_dim=256, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.backbone.set_grad_checkpointing(enable=True)
        
        # 冻结逻辑 (只训练最后几层)
        for param in self.backbone.parameters():
            param.requires_grad = False
        unfreeze_indices = [f"blocks.{i}." for i in range(37, 40)] + ["norm."]
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in unfreeze_indices):
                param.requires_grad = True

        # 投影层
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )
        
        # Confidence Head (学习率需设置大一点)
        self.confidence_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )        

    def forward(self, x):
        feat = self.backbone(x)  # 1536 维
        out_256 = self.proj(feat)
        confidence = self.confidence_head(feat) 
        # 返回: (投影特征, 置信度, 原始特征)
        return out_256, confidence, feat

# ==========================================
# 3. 带置信度的 DINO 版本
# ==========================================
@IMAGE_ENCODERS.register("vit_dino_with_confidence")
class ImageEncoderDINOWithConfidence(nn.Module):
    """
    全解冻 DINO 版本 (ViT-S/16)
    移除了所有冻结逻辑，允许整个 Backbone 参与对齐和微调
    """
    def __init__(self, model_name="vit_small_patch16_224_dino", pretrained=True, embed_dim=256, **kwargs):
        super().__init__()
        # 1. 加载 DINO
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # 2. 【全解冻】确保所有参数都参与梯度更新
        for param in self.backbone.parameters():
            param.requires_grad = True

        # 3. 投影层
        in_features = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )
        
        # 4. Confidence Head
        self.confidence_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )        

    def forward(self, x):
        # 提取全解冻的 DINO 特征
        raw_feat = self.backbone(x)
        
        # 投影到对齐空间
        out_256 = self.proj(raw_feat)
        
        # 计算置信度
        confidence = self.confidence_head(raw_feat)
        
        # 返回: (投影特征, 置信度, 原始特征)
        return out_256, confidence, raw_feat

# ==========================================
# 3. ViT DINO 版本
# ==========================================
@IMAGE_ENCODERS.register("vit_dino_v1")
class ImageEncoderViT(nn.Module):
    """
    对应 vit_small_patch16_224_dino 版本
    """
    def __init__(self, model_name="vit_small_patch16_224_dino", pretrained=True, embed_dim=256, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat


# ==========================================
# 4. MultiScale Wrapper (唯一版本 - 支持 config_dict)
# ==========================================
@IMAGE_ENCODERS.register("multiscale_image_encoder")
class MultiScaleImageEncoder(nn.Module):
    """
    ContextAwareImageEncoder - 支持 config_name 或 config_dict
    """
    def __init__(self, backbone_type="gigapath_frozen_v1", embed_dim=256, 
                 config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        # 动态调用 Backbone (传递 **kwargs 以忽略多余参数)
        self.image_encoder = IMAGE_ENCODERS.get(backbone_type)(embed_dim=embed_dim, **kwargs)
        
        # 支持两种配置方式: config_dict 优先，否则用 config_name
        self.context_processor = MultiScaleAwareEncoder(
            embed_dim, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict
        )

    def forward(self, target_img, context_imgs, target_pos, context_pos):
        batch_size = target_img.shape[0]
        
        # --- 1. 处理 Target ---
        target_res = self.image_encoder(target_img)
        
        # 检查是否返回了额外信息 (Tuple)
        target_conf, target_raw = None, None
        if isinstance(target_res, tuple):
            target_feat = target_res[0]
            target_conf = target_res[1]
            target_raw = target_res[2]
        else:
            target_feat = target_res
        
        # --- 2. 处理 Context (如果需要的话) ---
        if context_imgs.shape[1] == 0:
            return target_feat, target_conf, target_raw, target_feat
        
        num_context = context_imgs.shape[1]
        context_flat = context_imgs.reshape(-1, 3, context_imgs.shape[3], context_imgs.shape[4])
        
        context_res = self.image_encoder(context_flat)
        
        if isinstance(context_res, tuple):
            context_feats_flat = context_res[0]
        else:
            context_feats_flat = context_res
            
        context_feats = context_feats_flat.reshape(batch_size, num_context, -1)

        # --- 3. 空间融合 ---
        fused_output = self.context_processor(target_feat, context_feats, target_pos, context_pos)
        
        # 返回: (融合特征, 置信度, 原始特征, 局部特征作为Teacher)
        return fused_output, target_conf, target_raw, target_feat

# ==========================================
# 5. MultiScale Image Encoder with gate (支持 config_dict)
# ==========================================
@IMAGE_ENCODERS.register("gated_image_encoder")
class GatedImageEncoder(nn.Module):
    def __init__(self, backbone_type="vit_dino", embed_dim=256, 
                 config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        # 这里的 backbone 只输出特征，不带 confidence head
        self.image_encoder = IMAGE_ENCODERS.get(backbone_type)(embed_dim=embed_dim, **kwargs)
        
        self.context_processor = MultiScaleAwareEncoder(
            embed_dim, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict,
            use_gate=True
        )

    def forward(self, target_img, context_imgs, target_pos, context_pos):
        batch_size = target_img.shape[0]
        
        # 1. 局部特征 (Identity)
        target_feat = self.image_encoder(target_img) # 假设输出为 [B, E]
        
        # 2. 邻居特征 (Environment)
        if context_imgs.shape[1] == 0:
            return target_feat, torch.ones(batch_size, 1).to(target_feat.device)
        
        num_context = context_imgs.shape[1]
        context_flat = context_imgs.reshape(-1, 3, context_imgs.shape[3], context_imgs.shape[4])
        context_feats_flat = self.image_encoder(context_flat)
        context_feats = context_feats_flat.reshape(batch_size, num_context, -1)

        # 3. 门控融合
        # 内部公式: $$f_{final} = \alpha \cdot f_{local} + (1 - \alpha) \cdot f_{context}$$
        fused_output, alpha_img = self.context_processor(target_feat, context_feats, target_pos, context_pos)
        
        return fused_output, alpha_img
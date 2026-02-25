# src/models/image_encoders.py

import torch
import torch.nn as nn
import timm
from .registry import IMAGE_ENCODERS
from .multi_scale_blocks import MultiScaleAwareEncoder

# ==========================================
# 1. [[Model default]  Gated Image Encoder 
# ==========================================
@IMAGE_ENCODERS.register("gated_image_encoder")
class GatedImageEncoder(nn.Module):
    def __init__(self, backbone_type="vit_dino_v1", embed_dim=256, 
                 config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        # Backbone outputs features without a confidence head
        self.image_encoder = IMAGE_ENCODERS.get(backbone_type)(embed_dim=embed_dim, **kwargs)
        
        self.context_processor = MultiScaleAwareEncoder(
            embed_dim, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict,
            use_gate=True
        )

    def forward(self, target_img, context_imgs, target_pos, context_pos):
        batch_size = target_img.shape[0]
        
        # 1. Local Features (Identity)
        target_feat = self.image_encoder(target_img) # [B, E]
        
        # 2. Neighboring Features (Environment)
        if context_imgs.shape[1] == 0:
            return target_feat, torch.ones(batch_size, 1).to(target_feat.device)
        
        num_context = context_imgs.shape[1]
        context_flat = context_imgs.reshape(-1, 3, context_imgs.shape[3], context_imgs.shape[4])
        context_feats_flat = self.image_encoder(context_flat)
        context_feats = context_feats_flat.reshape(batch_size, num_context, -1)

        # 3. Gated Fusion
        # Internal formula: $f_{final} = \alpha \cdot f_{local} + (1 - \alpha) \cdot f_{context}$
        fused_output, alpha_img = self.context_processor(target_feat, context_feats, target_pos, context_pos)
        
        return fused_output, alpha_img
    
# ==========================================
# 2. Gigapath Frozen Version (optional)
# ==========================================
@IMAGE_ENCODERS.register("gigapath_frozen_v1")
class ImageEncoderGigaPath(nn.Module):
    """
    Standard ImageEncoder:
    - Backbone: GigaPath
    - Freeze all except last 3 blocks + norm
    - Proj: Linear -> LayerNorm -> GELU -> Linear
    """
    def __init__(self, model_name="hf_hub:prov-gigapath/prov-gigapath", pretrained=True, embed_dim=256, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.backbone.set_grad_checkpointing(enable=True)
        
        # Initial full freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze only the last 3 blocks + norm layer
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
# 3. GigaPath Encoder with Confidence Head
# ==========================================
@IMAGE_ENCODERS.register("gigapath_with_confidence")
class ImageEncoderWithConfidence(nn.Module):
    """
    Version with confidence_head for Distillation
    """
    def __init__(self, model_name="hf_hub:prov-gigapath/prov-gigapath", pretrained=True, embed_dim=256, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.backbone.set_grad_checkpointing(enable=True)
        
        # Freezing logic (training only last few layers)
        for param in self.backbone.parameters():
            param.requires_grad = False
        unfreeze_indices = [f"blocks.{i}." for i in range(37, 40)] + ["norm."]
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in unfreeze_indices):
                param.requires_grad = True

        # Projection Layer
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )
        
        # Confidence Head (Requires higher learning rate)
        self.confidence_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )        

    def forward(self, x):
        feat = self.backbone(x)  # 1536 dim
        out_256 = self.proj(feat)
        confidence = self.confidence_head(feat) 
        # Returns: (projected features, confidence, raw features)
        return out_256, confidence, feat

# ==========================================
# 4. DINO Version with Confidence Head
# ==========================================
@IMAGE_ENCODERS.register("vit_dino_with_confidence")
class ImageEncoderDINOWithConfidence(nn.Module):
    """
    Fully unfrozen DINO version (ViT-S/16).
    Allows the entire backbone to participate in alignment and fine-tuning.
    """
    def __init__(self, model_name="vit_small_patch16_224_dino", pretrained=True, embed_dim=256, **kwargs):
        super().__init__()
        # 1. Load DINO
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # 2. [Full Unfreeze] Ensure all parameters participate in gradient updates
        for param in self.backbone.parameters():
            param.requires_grad = True

        # 3. Projection Layer
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
        # Extract fully unfrozen DINO features
        raw_feat = self.backbone(x)
        
        # Project to alignment space
        out_256 = self.proj(raw_feat)
        
        # Calculate confidence
        confidence = self.confidence_head(raw_feat)
        
        # Returns: (projected features, confidence, raw features)
        return out_256, confidence, raw_feat

# ==========================================
#  ViT DINO Baseline
# ==========================================
@IMAGE_ENCODERS.register("vit_dino_v1")
class ImageEncoderViT(nn.Module):
    """
    Standard vit_small_patch16_224_dino implementation.
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
# MultiScale Wrapper (Supports config_dict)
# ==========================================
@IMAGE_ENCODERS.register("multiscale_image_encoder")
class MultiScaleImageEncoder(nn.Module):
    """
    ContextAwareImageEncoder - Supports config_name or config_dict.
    """
    def __init__(self, backbone_type="gigapath_frozen_v1", embed_dim=256, 
                 config_name="LongNet_for_spatial", config_dict=None, **kwargs):
        super().__init__()
        # Dynamically initialize backbone
        self.image_encoder = IMAGE_ENCODERS.get(backbone_type)(embed_dim=embed_dim, **kwargs)
        
        # Configuration priority: config_dict > config_name
        self.context_processor = MultiScaleAwareEncoder(
            embed_dim, embed_dim, 
            config_name=config_name, 
            config_dict=config_dict
        )

    def forward(self, target_img, context_imgs, target_pos, context_pos):
        batch_size = target_img.shape[0]
        
        # --- 1. Target Processing ---
        target_res = self.image_encoder(target_img)
        
        # Check for Tuple return (extra info)
        target_conf, target_raw = None, None
        if isinstance(target_res, tuple):
            target_feat = target_res[0]
            target_conf = target_res[1]
            target_raw = target_res[2]
        else:
            target_feat = target_res
        
        # --- 2. Context Processing ---
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

        # --- 3. Spatial Fusion ---
        fused_output = self.context_processor(target_feat, context_feats, target_pos, context_pos)
        
        # Returns: (fused features, confidence, raw features, local features as Teacher)
        return fused_output, target_conf, target_raw, target_feat


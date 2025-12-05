# longnet.py

import os, sys
import torch
import torch.nn as nn
import math

from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder, EncoderLayer
from torchscale.component.dilated_attention import DilatedAttention
from fairscale.nn import checkpoint_wrapper, wrap


############### version v1
# def twoD_sin_cos_emb(row, col, embed_dim=256, scale=1e4):
#     """
#     sin-Cos for 2D(row, col) pos-emb.
#     """
#     pe = torch.zeros(embed_dim, dtype=torch.float)
#     half_dim = embed_dim // 2
#     # row 的 sin-cos
#     for i in range(0, half_dim, 2):
#         div_term = scale ** (2*(i//2)/half_dim)  # i//2 而非 i
#         pe[i]   = math.sin(row / div_term)
#         pe[i+1] = math.cos(row / div_term)
#     # col 的 sin-cos
#     for i in range(0, half_dim, 2):
#         j = i + half_dim
#         div_term = scale ** (2*(i//2)/half_dim)
#         pe[j]   = math.sin(col / div_term)
#         pe[j+1] = math.cos(col / div_term)

#     return pe


########### Version v2
########### if change back please go back to this version
# def twoD_sin_cos_emb_batch(positions, embed_dim=256, scale=1e2, device=None):
#     """
#     Calculate 2D pos
    
#     positions: [batch_size, 2] or [batch_size, num_spots, 2]
#     """
#     if len(positions.shape) == 3:
#         # [batch_size, num_spots, 2]
#         batch_size, num_spots, _ = positions.shape
#         positions = positions.reshape(-1, 2)  # [batch_size*num_spots, 2]
#         reshape_back = True
#     else:
#         # [batch_size, 2]
#         batch_size = positions.shape[0]
#         num_spots = 1
#         reshape_back = False
    
#     # 创建位置编码张量 [batch_size*num_spots, embed_dim]
#     pe = torch.zeros(batch_size*num_spots, embed_dim, device=device or positions.device)
#     half_dim = embed_dim // 2
    
#     # 获取所有行列值
#     rows = positions[:, 0]
#     cols = positions[:, 1]
    
#     # 为所有行计算sin-cos编码
#     for i in range(0, half_dim, 2):
#         div_term = scale ** (2*(i//2)/half_dim)
#         pe[:, i] = torch.sin(rows / div_term)
#         pe[:, i+1] = torch.cos(rows / div_term)
    
#     # 为所有列计算sin-cos编码
#     for i in range(0, half_dim, 2):
#         j = i + half_dim
#         div_term = scale ** (2*(i//2)/half_dim)
#         pe[:, j] = torch.sin(cols / div_term)
#         pe[:, j+1] = torch.cos(cols / div_term)
    
#     if reshape_back:
#         pe = pe.reshape(batch_size, num_spots, embed_dim)
    
#     return pe

import torch
import math

def twoD_sin_cos_emb_batch(positions, embed_dim=256, device=None):
    """
    生成 2D 位置编码，遵循 ViT/Transformer 通用公式:
      PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
      PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )

    在2D场景下，我们将 embed_dim 分为两半:
      - 前 half_dim 用于 row 编码
      - 后 half_dim 用于 col 编码
    对 row/col 各自做 1D sin-cos 后拼接。
    
    参数:
      positions: [batch_size, 2] 或 [batch_size, n_spots, 2]，每个位置是 (row, col)
      embed_dim: 整体维度
      device   : 张量所在设备
    返回:
      pos_emb: [batch_size, n_spots, embed_dim] 或 [batch_size*n_spots, embed_dim]
               具体取决于输入是否需要 reshape_back
    """
    if len(positions.shape) == 3:
        # [B, N, 2]
        B, N, _ = positions.shape
        positions = positions.reshape(-1, 2)  # [B*N, 2]
        reshape_back = True
    else:
        # [B, 2]
        B = positions.shape[0]
        N = 1
        reshape_back = False

    device = device or positions.device

    # 最终输出大小: [B*N, embed_dim]
    pos_emb = torch.zeros((B*N, embed_dim), device=device)

    half_dim = embed_dim // 2
    # 取出行坐标/列坐标
    row_vals = positions[:, 0]  # [B*N]
    col_vals = positions[:, 1]  # [B*N]

    # 分别对 row/col 做 1D 的 sin-cos 编码，各占 half_dim
    row_emb = _get_1d_sin_cos_vit(row_vals, half_dim, device)
    col_emb = _get_1d_sin_cos_vit(col_vals, half_dim, device)

    # 拼接 => [B*N, half_dim + half_dim] = [B*N, embed_dim]
    pos_emb = torch.cat([row_emb, col_emb], dim=1)

    # 如果原输入是 3D，就 reshape 回去
    if reshape_back:
        pos_emb = pos_emb.reshape(B, N, embed_dim)

    return pos_emb


def _get_1d_sin_cos_vit(pos, dim, device=None):
    """
    对 1D 坐标用典型的 Transformer/ViT 公式做 sin-cos 编码:
      PE(pos,2i)   = sin( pos / 10000^(2i/d) )
      PE(pos,2i+1) = cos( pos / 10000^(2i/d) )
    返回 shape: [len(pos), dim]
    其中 dim 必须为偶数，以便一半做 sin，一半做 cos。
    """
    # 通常 dim 会是 128 之类（embed_dim 的一半）
    assert dim % 2 == 0, "dim must be even"

    # 生成 i 序列: 0,1,2,...(dim/2 - 1)
    half = dim // 2
    i = torch.arange(half, device=device).float()  # [dim/2]
    # 指数
    exponent = 2 * i / float(dim)  # [dim/2]
    # 频率项 freq: 1.0 / (10000^(2i/d))
    freq = 1.0 / (10000 ** exponent)  # [dim/2]

    # outer product: [len(pos), dim/2]
    # 每个 pos[i] 乘以 freq[j]
    pos_expand = pos.unsqueeze(1)  # [len(pos), 1]
    freq_expand = freq.unsqueeze(0)  # [1, dim/2]
    angle = pos_expand * freq_expand  # [len(pos), dim/2]

    # sin & cos
    sin_part = torch.sin(angle)  # [len(pos), dim/2]
    cos_part = torch.cos(angle)  # [len(pos), dim/2]

    emb = torch.cat([sin_part, cos_part], dim=1)  # [len(pos), dim]
    return emb

# ------------------------------
# (2) config dict
LongNet_8_layers_256_dim = {
    'encoder_layers': 8,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

# LongNetConfigs = {
#     "LongNet_8_layers_256_dim": LongNet_8_layers_256_dim
# }

###############################
# Visium
###############################
LongNet_for_spots = {
    'encoder_layers': 4,  
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 8, 
    'dilated_ratio': '[1, 2, 3, 4]', 
    'segment_length': '[1000, 2000, 3000, 5000]', 
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}
###############################
# Visium HD
###############################
LongNet_for_large_spatial = {
    'encoder_layers': 8,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16, 32]',  # 增加膨胀率以处理更长序列
    'segment_length': '[5000, 20000, 50000, 100000, 170000]',  # 适应更大的数据集
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNet_for_spatial = {
    'encoder_layers': 4,  
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 8,  
    'dilated_ratio': '[1, 2, 3, 4]',  
    'segment_length': '[1000, 2000, 3000, 5000]',  
    # 'segment_length': '[100, 200, 400, 800]',  
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

LongNetConfigs = {
    "LongNet_8_layers_256_dim": LongNet_8_layers_256_dim,
    "LongNet_for_spots": LongNet_for_spots,
    "LongNet_for_spatial": LongNet_for_spatial
}

# ------------------------------
# (3) LongNetEncoderLayer, LongNetEncoder 

class LongNetEncoderLayer(EncoderLayer):
    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

class LongNetEncoder(Encoder):
    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = LongNetEncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer




# ------------------------------
# (4) LongNetEncoder (DilatedAttention)
def make_longnet_from_name(config_name: str,
                           dilated_ratio: str='[1, 2, 3, 4]',  
                        #    segment_length: str='[100, 200, 400, 800]',
                           segment_length: str='[1000, 2000, 3000, 5000]',
                           drop_path_rate: float=0.1,
                           dropout: float=0.1):
    """
    Make a LongNetEncoder from a config name
    """
    if config_name not in LongNetConfigs:
        raise ValueError(f"Unknown config_name: {config_name}, please define in LongNetConfigs dict")

    longnet_args = LongNetConfigs[config_name].copy()  

    longnet_args['dropout'] = dropout
    longnet_args['drop_path_rate'] = drop_path_rate
    longnet_args['dilated_ratio'] = dilated_ratio
    longnet_args['segment_length'] = segment_length

    arch = EncoderConfig(**longnet_args)

    model = LongNetEncoder(arch)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Create LongNetEncoder: {config_name}, # trainable params= {n_params/1e6:.2f}M")
    return model
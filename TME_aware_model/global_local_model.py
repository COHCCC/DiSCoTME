import os
import time
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
from scipy.spatial import cKDTree

# 在 global_local_model.py 文件顶部
try:
    # 尝试相对导入（当作为包的一部分运行时）
    from .LongNet import make_longnet_from_name, twoD_sin_cos_emb_batch
except ImportError:
    # 失败则尝试直接导入（当在同一目录运行时）
    from LongNet import make_longnet_from_name, twoD_sin_cos_emb_batch
# ==============================
# Spot context 构建（VisiumHD 0522 v2.0 / Regular Visium 也适配）
# ==============================
def build_spot_contexts_fast(df, num_local, num_global, local_distance):
    coords = df[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values  # [N, 2]
    spot_ids = df['spot_id'].values                                   # [N,]
    N = len(df)

    tree = cKDTree(coords)

    spot_contexts = {}
    all_spots = list(spot_ids)

    for i, (target_id, target_pos) in enumerate(zip(spot_ids, coords)):
        # 1) local candidates (ball query) 并排除自身
        idxs_local_from_ball = tree.query_ball_point(target_pos, r=local_distance)
        local_candidates_indices = [idx for idx in idxs_local_from_ball if idx != i]

        # 2) 根据距离排序
        if not local_candidates_indices:
            sorted_pairs = []
        else:
            candidate_coords = coords[local_candidates_indices]
            local_dists = np.sqrt(np.sum((candidate_coords - target_pos) ** 2, axis=1))
            sorted_pairs = sorted(zip(local_candidates_indices, local_dists), key=lambda x: x[1])

        # 取 num_local，并在不足时做重复/回退填充（保持长度一致）
        if len(sorted_pairs) > num_local:
            local_spots = [spot_ids[idx] for idx, dist in sorted_pairs[:num_local]]
        else:
            local_spots = [spot_ids[idx] for idx, dist in sorted_pairs]
            if num_local > 0 and len(local_spots) < num_local:
                if len(local_spots) > 0:
                    repeats_needed = num_local - len(local_spots)
                    local_spots.extend(random.choices(local_spots, k=repeats_needed))
                else:
                    other_spots_ids = [s_id for s_id in all_spots if s_id != target_id]
                    if other_spots_ids:
                        local_spots = random.sample(other_spots_ids, min(num_local, len(other_spots_ids)))
                    else:
                        local_spots = []

        # 3) global spots（只在 num_global > 0 时采样）
        global_spots = []
        local_set = set(local_spots)
        remaining_spots_ids = [s_id for s_id in all_spots if s_id not in local_set and s_id != target_id]

        if num_global > 0:
            if len(remaining_spots_ids) > num_global:
                global_spots = random.sample(remaining_spots_ids, num_global)
            else:
                global_spots = remaining_spots_ids.copy()
                if 0 < len(global_spots) < num_global:
                    repeats_needed = num_global - len(global_spots)
                    global_spots.extend(random.choices(global_spots, k=repeats_needed))
                elif len(global_spots) == 0:
                    other_spots_ids_for_global = [s_id for s_id in all_spots if s_id != target_id]
                    if other_spots_ids_for_global:
                        global_spots = random.sample(
                            other_spots_ids_for_global, min(num_global, len(other_spots_ids_for_global))
                        )

        spot_contexts[target_id] = {
            'local': local_spots,
            'global': global_spots
        }

    return spot_contexts


# ==============================
# Dataset（Regular Visium & VisiumHD 均可）
# ==============================
class GlobalLocalContextDataset(Dataset):
    def __init__(self,
                 metadata_csv,
                 tissue_positions_csv,
                 root_dir,
                 transform_image=None,
                 num_local=10,
                 num_global=30,
                 local_distance=500):
        self.root_dir = root_dir
        self.transform_image = transform_image
        self.num_local = num_local
        self.num_global = num_global
        self.local_distance = local_distance

        is_rank_0 = (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0)

        if is_rank_0:
            print(
                f"Dataset Initializing with num_local={self.num_local}, "
                f"num_global={self.num_global}, local_distance={self.local_distance}",
                flush=True
            )
            overall_init_start_time = time.time()

        # 2) 读取元数据
        if is_rank_0: t_start = time.time()
        meta_path = os.path.join(self.root_dir, metadata_csv)
        self.df_meta = pd.read_csv(meta_path)
        if is_rank_0: print(f"Time to load df_meta: {time.time() - t_start:.2f}s", flush=True)

        # 3) 读取空间位置数据（.parquet 视为 HD，否则视为 regular Visium CSV）
        if is_rank_0: t_start = time.time()
        if tissue_positions_csv.endswith('.parquet'):
            import pyarrow.parquet as pq
            pos_path = os.path.join(
                "/coh_labs/dits/nsong/Mayo_VisiumHD/101/outs", "binned_outputs", "square_016um", "spatial",
                tissue_positions_csv
            )
            self.df_spatial = pq.read_table(pos_path).to_pandas()
            self.df_spatial.rename(columns={"barcode": "spot_id"}, inplace=True)
        else:
            pos_path = os.path.join(self.root_dir, "spatial", tissue_positions_csv)
            self.df_spatial = pd.read_csv(pos_path, header=None)
            self.df_spatial.columns = [
                "spot_id", "in_tissue", "row", "col", "pxl_row_in_fullres", "pxl_col_in_fullres"
            ]
        if is_rank_0: print(f"Time to load df_spatial: {time.time() - t_start:.2f}s", flush=True)

        # 4) 合并
        if is_rank_0: t_start = time.time()
        self.df = pd.merge(self.df_meta, self.df_spatial, on="spot_id", how="inner")
        if is_rank_0: print(f"Time to merge dataframes: {time.time() - t_start:.2f}s", flush=True)

        if is_rank_0:
            print(f"Dataset contains {len(self.df)} valid spots after merge.", flush=True)

        # 5) 坐标处理与归一化
        if is_rank_0: t_start = time.time()
        self.df['pxl_row_in_fullres'] = pd.to_numeric(self.df['pxl_row_in_fullres'], errors='coerce')
        self.df['pxl_col_in_fullres'] = pd.to_numeric(self.df['pxl_col_in_fullres'], errors='coerce')
        self.df.dropna(subset=['pxl_row_in_fullres', 'pxl_col_in_fullres'], inplace=True)

        if len(self.df) == 0 and is_rank_0:
            print("WARNING: DataFrame is empty after dropping NaNs from coordinate columns. Check data processing.",
                  flush=True)

        if len(self.df) > 0:
            self.row_min, self.row_max = self.df['pxl_row_in_fullres'].min(), self.df['pxl_row_in_fullres'].max()
            self.col_min, self.col_max = self.df['pxl_col_in_fullres'].min(), self.df['pxl_col_in_fullres'].max()

            if (self.row_max - self.row_min) == 0:
                self.df['norm_row'] = 0.0
            else:
                self.df['norm_row'] = (self.df['pxl_row_in_fullres'] - self.row_min) / (self.row_max - self.row_min) * 10

            if (self.col_max - self.col_min) == 0:
                self.df['norm_col'] = 0.0
            else:
                self.df['norm_col'] = (self.df['pxl_col_in_fullres'] - self.col_min) / (self.col_max - self.col_min) * 10
        else:
            # 空 df 时兜底
            self.df['norm_row'] = 0.0
            self.df['norm_col'] = 0.0

        if is_rank_0: print(f"Time for df column processing and normalization: {time.time() - t_start:.2f}s", flush=True)

        # 6) 构建上下文
        if is_rank_0:
            print("Building spot contexts using (V1 RegularVisium) build_spot_contexts_fast...", flush=True)
            build_contexts_start_time = time.time()

        self.spot_contexts = build_spot_contexts_fast(self.df, self.num_local, self.num_global, self.local_distance)

        if is_rank_0:
            print(
                f"Context building (V1) complete. Each spot has {self.num_local} local and {self.num_global} global "
                f"context spots. Time taken: {time.time() - build_contexts_start_time:.2f}s", flush=True
            )
            print(f"Total Dataset Initialization time: {time.time() - overall_init_start_time:.2f}s", flush=True)

    def __len__(self):
        return len(self.df) if hasattr(self, 'df') else 0

    def _load_image(self, row):
        img_filename = row["image_path"]
        img_path = os.path.join(self.root_dir, img_filename)

        if img_filename.lower().endswith(".npy"):
            try:
                img_array = np.load(img_path)
                if img_array.shape[-1] == 3 and img_array.dtype == np.uint8:
                    img = Image.fromarray(img_array, 'RGB')
                else:
                    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
                        print(f"Warning: Unexpected .npy format. Path: {img_path}, Shape: {img_array.shape}, "
                              f"Dtype: {img_array.dtype}. Attempting direct fromarray.")
                    img = Image.fromarray(img_array)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
            except Exception as e:
                if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
                    print(f"Error loading .npy file {img_path}: {e}", flush=True)
                raise
        else:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
                    print(f"Error loading non-npy image file {img_path}: {e}", flush=True)
                raise

        if self.transform_image:
            img = self.transform_image(img)
        return img

    def _load_gene(self, row):
        gene_path = os.path.join(self.root_dir, row["gene_vector_path"])
        gene_vec = np.load(gene_path)
        return torch.tensor(gene_vec, dtype=torch.float32)

    def __getitem__(self, idx):
        target_row = self.df.iloc[idx]
        target_id = target_row['spot_id']

        target_img = self._load_image(target_row)
        target_gene = self._load_gene(target_row)
        target_pos = torch.tensor([target_row['norm_row'], target_row['norm_col']], dtype=torch.float32)

        local_ids = self.spot_contexts[target_id]['local']
        global_ids = self.spot_contexts[target_id]['global']
        context_ids = (local_ids if isinstance(local_ids, list) else list(local_ids)) + \
                      (global_ids if isinstance(global_ids, list) else list(global_ids))

        local_len = len(local_ids) if isinstance(local_ids, list) else 0
        global_len = len(global_ids) if isinstance(global_ids, list) else 0

        context_imgs, context_genes, context_pos_list = [], [], []

        if context_ids:
            for context_id in context_ids:
                try:
                    context_row_series = self.df[self.df['spot_id'] == context_id].iloc[0]
                except IndexError:
                    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
                        print(f"WARNING: Context ID {context_id} not found in self.df for target_id {target_id}. "
                              f"Skipping this context spot.", flush=True)
                    continue

                context_imgs.append(self._load_image(context_row_series))
                context_genes.append(self._load_gene(context_row_series))
                context_pos_list.append([context_row_series['norm_row'], context_row_series['norm_col']])

        if not context_imgs:
            num_context_elements = self.num_local + self.num_global
            if isinstance(target_img, torch.Tensor):
                context_imgs_tensor = torch.empty((0, *target_img.shape))
            else:
                context_imgs_tensor = torch.empty((0, 3, 224, 224))

            if isinstance(target_gene, torch.Tensor):
                context_genes_tensor = torch.empty((0, *target_gene.shape))
            else:
                context_genes_tensor = torch.empty((0, 2000))

            context_pos_tensor = torch.empty((0, 2))
        else:
            context_imgs_tensor = torch.stack(context_imgs)
            context_genes_tensor = torch.stack(context_genes)
            context_pos_tensor = torch.tensor(context_pos_list, dtype=torch.float32)

        return {
            'target_img': target_img,
            'target_gene': target_gene,
            'target_pos': target_pos,
            'context_imgs': context_imgs_tensor,
            'context_genes': context_genes_tensor,
            'context_pos': context_pos_tensor,
            'target_id': target_id,
            'local_len': torch.tensor(local_len, dtype=torch.int),
            'global_len': torch.tensor(global_len, dtype=torch.int),
        }


# ==============================
# 模型组件（保持你提供的结构）
# ==============================
class GlobalLocalAwareEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256, config_name="LongNet_for_spatial"):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.longnet = make_longnet_from_name(config_name)

    def forward(self, target_feat, context_feats, target_pos, context_pos):
        """
        Using LongNet to capture global and local context

        target_feat:   [batch_size, input_dim]
        context_feats: [batch_size, num_context, input_dim]
        target_pos:    [batch_size, 2]
        context_pos:   [batch_size, num_context, 2]
        """
        batch_size = target_feat.shape[0]
        num_context = context_feats.shape[1]

        target_proj = self.input_proj(target_feat)  # [B, E]

        context_flat = context_feats.reshape(-1, self.input_dim)
        context_proj = self.input_proj(context_flat)
        context_proj = context_proj.reshape(batch_size, num_context, self.embed_dim)

        target_pos_emb = twoD_sin_cos_emb_batch(target_pos, self.embed_dim, device=target_proj.device)
        target_proj = target_proj + 0.3 * target_pos_emb

        context_pos_emb = twoD_sin_cos_emb_batch(context_pos, self.embed_dim, device=context_proj.device)
        context_proj = context_proj + 0.3 * context_pos_emb

        target_proj = target_proj.unsqueeze(1)                           # [B, 1, E]
        combined = torch.cat([target_proj, context_proj], dim=1)         # [B, 1+C, E]

        out_dict = self.longnet(src_tokens=None, token_embeddings=combined)
        seq_output = out_dict["encoder_out"]                             # [B, 1+C, E]

        target_output = seq_output[:, 0]                                 # [B, E]
        return target_output


class ImageEncoder(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224_dino", pretrained=True, embed_dim=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        feat = self.backbone(x)  # [B, in_features]
        feat = self.proj(feat)   # [B, E]
        return feat


class ContextAwareImageEncoder(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224_dino", pretrained=True, embed_dim=256, config_name="LongNet_for_spatial"):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name, pretrained, embed_dim)
        self.context_processor = GlobalLocalAwareEncoder(embed_dim, embed_dim, config_name)

    def forward(self, target_img, context_imgs, target_pos, context_pos):
        """
        target_img:   [batch_size, 3, H, W]
        context_imgs: [batch_size, num_context, 3, H, W]
        """
        batch_size = target_img.shape[0]
        num_context = context_imgs.shape[1]

        target_feat = self.image_encoder(target_img)  # [B, E]

        context_flat = context_imgs.reshape(-1, 3, context_imgs.shape[3], context_imgs.shape[4])
        context_feats_flat = self.image_encoder(context_flat)            # [B*C, E]
        context_feats = context_feats_flat.reshape(batch_size, num_context, -1)  # [B, C, E]

        output = self.context_processor(target_feat, context_feats, target_pos, context_pos)
        return output

# 最原版
# class ContextAwareGeneEncoder(nn.Module):
#     # 按你的原始结构保留 input_proj（虽然 forward 未使用）
#     def __init__(self, gene_dim=2000, embed_dim=256, config_name="LongNet_for_spatial"):
#         super().__init__()
#         self.input_proj = nn.Linear(gene_dim, embed_dim)
#         self.context_processor = GlobalLocalAwareEncoder(gene_dim, embed_dim, config_name)

#     def forward(self, target_gene, context_genes, target_pos, context_pos):
#         return self.context_processor(target_gene, context_genes, target_pos, context_pos)

# 改进版：先做一次linear降维到256在MHA，再送入 LongNet
# class ContextAwareGeneEncoder(nn.Module):
#     def __init__(self, gene_dim=2000, embed_dim=256, num_heads=4, config_name="LongNet_for_spatial"):
#         super().__init__()
#         # gene 特征线性投影
#         self.input_proj = nn.Linear(gene_dim, embed_dim)
#         # 单层 self-attention
#         self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         # Global/Local context processor
#         self.context_processor = GlobalLocalAwareEncoder(embed_dim, embed_dim, config_name)

#     def forward(self, target_gene, context_genes, target_pos, context_pos):
#         # gene vector -> [B, 1, E]
#         target_gene = self.input_proj(target_gene).unsqueeze(1)
#         context_genes = self.input_proj(context_genes)       # [B, C, E]

#         # 简单拼接 target+context，做一次 attention
#         combined = torch.cat([target_gene, context_genes], dim=1)  # [B, 1+C, E]
#         attn_out, _ = self.attn(combined, combined, combined)      # [B, 1+C, E]

#         # 取 target 的输出
#         target_out = attn_out[:, 0, :]   # [B, E]
#         return self.context_processor(target_out, context_genes, target_pos, context_pos)


# # 改进版2: 直接在2000维空间做 MHA再linear，再降维到256
# class ContextAwareGeneEncoder(nn.Module):
#     def __init__(self, gene_dim=2000, embed_dim=256, num_heads=4, config_name="LongNet_for_spatial"):
#         super().__init__()
#         # 直接在 2000 维空间做 MultiheadAttention
#         self.attn = nn.MultiheadAttention(embed_dim=gene_dim, num_heads=num_heads, batch_first=True)
#         # Attention 输出 (2000) → 降到 256
#         self.down_proj = nn.Linear(gene_dim, embed_dim)
#         # Global/Local context processor
#         self.context_processor = GlobalLocalAwareEncoder(embed_dim, embed_dim, config_name)

#     def forward(self, target_gene, context_genes, target_pos, context_pos):
#         # 输入 [B, 2000]
#         target_gene = target_gene.unsqueeze(1)   # [B, 1, 2000]
#         context_genes = context_genes            # [B, C, 2000]

#         # 拼接 target + context → Attention
#         combined = torch.cat([target_gene, context_genes], dim=1)  # [B, 1+C, 2000]
#         attn_out, _ = self.attn(combined, combined, combined)      # [B, 1+C, 2000]

#         # 取 target 的输出并降维
#         target_out = attn_out[:, 0, :]           # [B, 2000]
#         target_out = self.down_proj(target_out)  # [B, 256]

#         # 下游 global/local context
#         return self.context_processor(target_out, 
#                                       self.down_proj(context_genes),  # context 也降到 256
#                                       target_pos, context_pos)    
    
# # 改进版3: 直接在2000维空间做 MHA不降维   （目前最好的版本）
# class ContextAwareGeneEncoder(nn.Module):
#     def __init__(self, gene_dim=2000, embed_dim=256, num_heads=4, config_name="LongNet_for_spatial"):
#         super().__init__()
#         # 2000维里做 MHA
#         assert gene_dim % num_heads == 0, "embed_dim (gene_dim) must be divisible by num_heads"
#         self.attn = nn.MultiheadAttention(embed_dim=gene_dim, num_heads=num_heads, batch_first=True)
#         # 2000 -> 256
#         self.down_proj = nn.Linear(gene_dim, embed_dim)
#         # LongNet 在 256 维工作
#         self.context_processor = GlobalLocalAwareEncoder(embed_dim, embed_dim, config_name)

#     def forward(self, target_gene, context_genes, target_pos, context_pos):
#         # target_gene: [B, 2000] -> [B, 1, 2000]
#         target_gene = target_gene.unsqueeze(1)                         # [B, 1, 2000]
#         combined = torch.cat([target_gene, context_genes], dim=1)      # [B, 1+C, 2000]

#         # 在 2000 维做注意力
#         attn_out, _ = self.attn(combined, combined, combined, need_weights=False)  # [B, 1+C, 2000]

#         # target / context 都用注意力后的表征再降维到 256
#         target_out  = self.down_proj(attn_out[:, 0, :])    # [B, 256]
#         context_out = self.down_proj(attn_out[:, 1:, :])   # [B, C, 256]

#         # 进入 LongNet（256维）
#         return self.context_processor(target_out, context_out, target_pos, context_pos)

 
# 改进版4: 先在2000维做self-attention，再降维到256，再送入 LongNet
# class ContextAwareGeneEncoder(nn.Module):
#     def __init__(self, gene_dim=2000, embed_dim=256, num_heads=8, config_name="LongNet_for_spatial"):
#         super().__init__()
        
#         # 直接在2000维做self-attention
#         assert gene_dim % num_heads == 0
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=gene_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
        
#         # 降维到embed_dim
#         self.proj = nn.Linear(gene_dim, embed_dim)
        
#         # Context处理器
#         self.context_processor = GlobalLocalAwareEncoder(embed_dim, embed_dim, config_name)
    
#     def encode_genes(self, genes):
#         """独立的基因编码器"""
#         # Self-attention in high-dim space
#         x = genes.unsqueeze(1)  # [B, 1, 2000]
#         x, _ = self.self_attn(x, x, x)
#         x = x.squeeze(1)  # [B, 2000]
        
#         # Project to embed_dim
#         return self.proj(x)  # [B, 256]
    
#     def forward(self, target_gene, context_genes, target_pos, context_pos):
#         # 独立编码
#         target_feat = self.encode_genes(target_gene)
        
#         B, C, D = context_genes.shape
#         context_flat = context_genes.reshape(B*C, D)
#         context_feats_flat = self.encode_genes(context_flat)
#         context_feats = context_feats_flat.reshape(B, C, -1)
        
#         # LongNet融合
#         return self.context_processor(target_feat, context_feats, target_pos, context_pos)

# 改进版5: 先在2000维做TransformerEncoder，再降维到256，再送入 LongNet
# class ContextAwareGeneEncoder(nn.Module):
#     def __init__(self, gene_dim=2000, embed_dim=256, num_heads=8, 
#                  num_layers=2, config_name="LongNet_for_spatial"):
#         super().__init__()
        
#         # 在2000维做transformer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=gene_dim,
#             nhead=num_heads,
#             batch_first=True,
#             dim_feedforward=2048,   # FFN的hidden size，可以调
#             dropout=0.1
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # 降维到embed_dim
#         self.proj = nn.Linear(gene_dim, embed_dim)
        
#         # Context处理器 (256维空间)
#         self.context_processor = GlobalLocalAwareEncoder(embed_dim, embed_dim, config_name)
    
#     def encode_genes(self, genes):
#         # 输入 [B, D] → [B, 1, D]
#         x = genes.unsqueeze(1)
#         x = self.transformer(x)   # [B, 1, D]
#         x = x.squeeze(1)          # [B, D]
#         return self.proj(x)       # [B, 256]
    
#     def forward(self, target_gene, context_genes, target_pos, context_pos):
#         # target
#         target_feat = self.encode_genes(target_gene)
        
#         # context
#         B, C, D = context_genes.shape
#         context_flat = context_genes.reshape(B*C, D)
#         context_feats_flat = self.encode_genes(context_flat)
#         context_feats = context_feats_flat.reshape(B, C, -1)
        
#         # LongNet融合
#         return self.context_processor(target_feat, context_feats, target_pos, context_pos)

# 版本6: 直接在2000维做TransformerEncoder，再送入 LongNet（目前在spa1上最make sense）
class ContextAwareGeneEncoder(nn.Module):
    def __init__(self, gene_dim=2000, embed_dim=256, num_heads=8, config_name="LongNet_for_spatial"):
        super().__init__()
        
        # 2000维里做 TransformerEncoder (多层MHA)
        assert gene_dim % num_heads == 0, "gene_dim must be divisible by num_heads"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gene_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=2048,  # 可以调大一些
            dropout=0.1,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 1层，可调多层
        
        # LongNet 在 2000维工作
        self.context_processor = GlobalLocalAwareEncoder(gene_dim, embed_dim, config_name)

    def forward(self, target_gene, context_genes, target_pos, context_pos):
        # target_gene: [B, 2000] -> [B, 1, 2000]
        target_gene = target_gene.unsqueeze(1)  # [B, 1, 2000]
        combined = torch.cat([target_gene, context_genes], dim=1)  # [B, 1+C, 2000]

        # Transformer编码 (spot内部 self-attention)
        out = self.transformer(combined)  # [B, 1+C, 2000]

        target_out = out[:, 0, :]   # [B, 2000]
        context_out = out[:, 1:, :] # [B, C, 2000]

        # LongNet 融合 (全程 2000维)
        return self.context_processor(target_out, context_out, target_pos, context_pos)


class GlobalLocalMultiModalModel(nn.Module):
    def __init__(self, img_encoder, gene_encoder, proj_dim=128):
        super().__init__()
        self.img_encoder = img_encoder
        self.gene_encoder = gene_encoder

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

    def get_image_local_embedding(self, target_img, context_imgs, target_pos, context_pos):
        img_feat = self.img_encoder(target_img, context_imgs, target_pos, context_pos)
        img_emb = self.img_proj(img_feat)
        return img_emb

    def forward(self, batch):
        """
        batch keys:
            - target_img, target_gene, target_pos
            - context_imgs, context_genes, context_pos
        """
        target_img = batch['target_img']
        target_gene = batch['target_gene']
        target_pos = batch['target_pos']
        context_imgs = batch['context_imgs']
        context_genes = batch['context_genes']
        context_pos = batch['context_pos']

        img_feat = self.img_encoder(target_img, context_imgs, target_pos, context_pos)
        gene_feat = self.gene_encoder(target_gene, context_genes, target_pos, context_pos)

        img_emb = self.img_proj(img_feat)
        gene_emb = self.gene_proj(gene_feat)

        return img_emb, gene_emb
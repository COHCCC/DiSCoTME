# src/data/dataset.py

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image
from src.utils.context_utils import build_spot_contexts_fast # Import context builder

class MultiScaleContextDataset(Dataset):
    def __init__(self,
                 metadata_csv,
                 tissue_positions_csv,
                 root_dir,
                 transform_image=None,
                 num_local=35,
                 num_global=0,
                 local_distance=512):
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

        # 3) 读取空间位置数据
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
            print("WARNING: DataFrame is empty after dropping NaNs from coordinate columns.", flush=True)

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
            self.df['norm_row'] = 0.0
            self.df['norm_col'] = 0.0

        if is_rank_0: print(f"Time for df column processing and normalization: {time.time() - t_start:.2f}s", flush=True)

        # 6) 构建上下文
        if is_rank_0:
            print("Building spot contexts using build_spot_contexts_fast...", flush=True)
            build_contexts_start_time = time.time()

        self.spot_contexts = build_spot_contexts_fast(self.df, self.num_local, self.num_global, self.local_distance)

        if is_rank_0:
            print(
                f"Context building complete. Time taken: {time.time() - build_contexts_start_time:.2f}s", flush=True
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
                        print(f"Warning: Unexpected .npy format. Path: {img_path}")
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
                        print(f"WARNING: Context ID {context_id} not found", flush=True)
                    continue

                context_imgs.append(self._load_image(context_row_series))
                context_genes.append(self._load_gene(context_row_series))
                context_pos_list.append([context_row_series['norm_row'], context_row_series['norm_col']])

        if not context_imgs:
            if isinstance(target_img, torch.Tensor):
                context_imgs_tensor = torch.empty((0, *target_img.shape))
            else:
                context_imgs_tensor = torch.empty((0, 3, 224, 224))

            if isinstance(target_gene, torch.Tensor):
                context_genes_tensor = torch.empty((0, *target_gene.shape))
            else:
                context_genes_tensor = torch.empty((0, 512))

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
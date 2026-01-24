#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc
import tifffile

# --- 工具函数 (保持不变) ---

def read_scalefactors(root_dir: str) -> dict:
    cands = [os.path.join(root_dir, "spatial", "scalefactors_json.json"),
             os.path.join(root_dir, "scalefactors_json.json")]
    for p in cands:
        if os.path.exists(p):
            with open(p, "r") as f: return json.load(f)
    raise FileNotFoundError(f"Missing scalefactors_json.json in {root_dir}")

def infer_radius_from_scalefactors(root_dir: str) -> int:
    sf = read_scalefactors(root_dir)
    return max(int(math.floor(float(sf["spot_diameter_fullres"]) / 2.0)), 1)

def find_tissue_csv(root_dir: str) -> str:
    cands = [os.path.join(root_dir, "spatial", "tissue_positions.csv"),
             os.path.join(root_dir, "spatial", "tissue_positions_list.csv")]
    for p in cands:
        if os.path.exists(p): return p
    raise FileNotFoundError("Missing tissue_positions.csv")

def autodetect_wsi(root_dir):
    base = os.path.basename(os.path.normpath(root_dir))
    for ext in (".tif", ".tiff"):
        p = os.path.join(root_dir, base + ext)
        if os.path.exists(p): return p
    hits = sorted(glob.glob(os.path.join(root_dir, "*.tif"))) + \
           sorted(glob.glob(os.path.join(root_dir, "*.tiff")))
    if hits: return hits[0]
    raise FileNotFoundError(f"No WSI found in {root_dir}")

def load_positions(spatial_csv: str):
    rows = []
    with open(spatial_csv, 'r') as f:
        first = f.readline().strip()
        has_header = "barcode" in first or "in_tissue" in first
        if not has_header and first:
            fields = first.split(',')
            if len(fields) >= 6 and float(fields[1]) == 1:
                rows.append((fields[0], int(float(fields[5])), int(float(fields[4]))))
        for line in f:
            fields = line.strip().split(',')
            if float(fields[1]) == 1:
                rows.append((fields[0], int(float(fields[5])), int(float(fields[4]))))
    return rows

def load_wsi_into_memory(tif_path: str):
    print(f"Loading WSI: {tif_path}")
    with tifffile.TiffFile(tif_path) as tif:
        arr = tif.asarray()
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    return arr

def _normalize_rgb(arr):
    if arr is None: return None
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 4: arr = arr[..., :3]
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else a * 0
        arr = a.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")

# --- 核心逻辑 ---

def produce_test_data(root_dir, wsi_path, radius, out_dir, hvg_method, n_genes):
    os.makedirs(os.path.join(out_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "genes"), exist_ok=True)

    # 1. 加载 WSI 和坐标
    wsi_array = load_wsi_into_memory(wsi_path)
    rows = load_positions(find_tissue_csv(root_dir))

    # 2. 加载并处理表达谱
    h5_file = os.path.join(root_dir, "filtered_feature_bc_matrix.h5")
    adata = sc.read_10x_h5(h5_file)
    adata.var_names_make_unique()
    
    # 过滤线粒体基因
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata = adata[:, ~adata.var["mt"]]
    sc.pp.filter_genes(adata, min_cells=10)

    # 【你的核心逻辑】备份 raw counts
    adata.layers["counts"] = adata.X.copy()

    # 先做标准归一化/对数化 (用于 adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 【用户二选一】HVG 选择
    if hvg_method == "seurat_v3":
        print(f"--> Using seurat_v3 (from layers['counts']) to find {n_genes} HVGs")
        sc.pp.highly_variable_genes(
            adata, 
            flavor="seurat_v3", 
            n_top_genes=n_genes, 
            layer="counts", 
            inplace=True
        )
    else:
        print(f"--> Using default method (from log-normalized adata.X) to find {n_genes} HVGs")
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_genes, 
            inplace=True
        )

    # 筛选基因
    adata = adata[:, adata.var['highly_variable']]
    print(f"Final data shape: {adata.shape}")

    # 3. 切片与保存
    metadata = []
    for idx, (barcode, x, y) in enumerate(rows):
        if barcode not in adata.obs.index: continue
        
        # 保存图片
        x0, y0 = int(x - radius), int(y - radius)
        x1, y1 = int(x + radius), int(y + radius)
        patch = wsi_array[y0:y1, x0:x1]
        _normalize_rgb(patch).save(os.path.join(out_dir, "image", f"{barcode}.jpg"))

        # 保存基因 (确保转为 dense)
        g_data = adata[barcode, :].X
        if hasattr(g_data, "toarray"): g_data = g_data.toarray()
        np.save(os.path.join(out_dir, "genes", f"{barcode}.npy"), g_data.astype(np.float32).flatten())

        metadata.append({"spot_id": barcode, "image": f"image/{barcode}.jpg", "gene": f"genes/{barcode}.npy"})

    pd.DataFrame(metadata).to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
    print("Done!")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="SpaceRanger output folder")
    p.add_argument("--wsi", default=None)
    p.add_argument("--radius", type=int, default=None)
    p.add_argument("--out", default=None)
    # 二选一参数
    p.add_argument("--hvg_method", choices=["default", "seurat_v3"], default="default", help="HVG selection flavor")
    p.add_argument("--n_genes", type=int, default=2000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    rd = os.path.abspath(args.root)
    produce_test_data(
        rd, 
        os.path.abspath(args.wsi) if args.wsi else autodetect_wsi(rd),
        args.radius if args.radius else infer_radius_from_scalefactors(rd),
        os.path.abspath(args.out) if args.out else rd,
        args.hvg_method,
        args.n_genes
    )
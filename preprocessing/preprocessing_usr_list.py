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

def read_scalefactors(root_dir: str) -> dict:
    cands = [os.path.join(root_dir, "spatial", "scalefactors_json.json"),
             os.path.join(root_dir, "scalefactors_json.json")]
    for p in cands:
        if os.path.exists(p):
            with open(p, "r") as f: return json.load(f)
    raise FileNotFoundError(f"Cannot find scalefactors_json.json under {root_dir}.")

def infer_radius_from_scalefactors(root_dir: str) -> int:
    sf = read_scalefactors(root_dir)
    return max(int(math.floor(float(sf["spot_diameter_fullres"]) / 2.0)), 1)

def find_tissue_csv(root_dir: str) -> str:
    cands = [os.path.join(root_dir, "spatial", "tissue_positions.csv"),
             os.path.join(root_dir, "spatial", "tissue_positions_list.csv")]
    for p in cands:
        if os.path.exists(p): return p
    raise FileNotFoundError("Cannot find tissue_positions csv under spatial/.")

def autodetect_wsi(root_dir):
    base = os.path.basename(os.path.normpath(root_dir))
    for ext in (".tif", ".tiff"):
        p = os.path.join(root_dir, base + ext)
        if os.path.exists(p): return p
    hits = sorted(glob.glob(os.path.join(root_dir, "*.tif"))) + \
           sorted(glob.glob(os.path.join(root_dir, "*.tiff")))
    if hits: return hits[0]
    raise FileNotFoundError(f"No WSI (.tif) found in {root_dir}.")

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
            if len(fields) >= 6 and float(fields[1]) == 1:
                rows.append((fields[0], int(float(fields[5])), int(float(fields[4]))))
    return rows

def _normalize_rgb(arr):
    if arr is None: return None
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 4: arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        a = arr.astype(np.float32)
        lo, hi = a.min(), a.max()
        arr = ((a - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(a, dtype=np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.ndim == 3:
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            lo, hi = a.min(), a.max()
            arr = ((a - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(a, dtype=np.uint8)
        img = Image.fromarray(arr)
        return img.convert("RGB") if img.mode != "RGB" else img
    return None

def load_wsi_into_memory(tif_path: str):
    print(f"Loading WSI: {tif_path}")
    with tifffile.TiffFile(tif_path) as tif:
        wsi_array = tif.asarray()
    if wsi_array.ndim == 3 and wsi_array.shape[0] in (3, 4) and wsi_array.shape[0] < wsi_array.shape[-1]:
        wsi_array = np.moveaxis(wsi_array, 0, -1)
    return wsi_array

def read_patch_from_array(wsi_array, x_center, y_center, radius):
    H, W = wsi_array.shape[:2]
    x0, y0 = max(0, int(round(x_center - radius))), max(0, int(round(y_center - radius)))
    x1, y1 = min(W, int(round(x_center + radius))), min(H, int(round(y_center + radius)))
    if x1 <= x0 or y1 <= y0: return None
    return _normalize_rgb(wsi_array[y0:y1, x0:x1])

def produce_test_data(root_dir, wsi_path, radius, test_data_dir, gene_list_path=None):
    spatial_csv = find_tissue_csv(root_dir)
    h5_file = os.path.join(root_dir, "filtered_feature_bc_matrix.h5")

    out_image_dir = os.path.join(test_data_dir, "image")
    out_genes_dir = os.path.join(test_data_dir, "genes")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_genes_dir, exist_ok=True)

    wsi_array = load_wsi_into_memory(wsi_path)
    rows = load_positions(spatial_csv)

    print(f"Loading {h5_file}...")
    adata = sc.read_10x_h5(h5_file)
    adata.var_names_make_unique()

    if gene_list_path and os.path.exists(gene_list_path):
        print(f"Applying gene list from {gene_list_path}...")
        gene_df = pd.read_csv(gene_list_path)
        ranked_genes = gene_df["Name"].astype(str).tolist()
        valid_genes = [g for g in ranked_genes if g in adata.var_names]
        adata = adata[:, valid_genes].copy()
        print(f"Kept {len(valid_genes)} genes in specified order.")
    else:
        print("Warning: gene list not provided, using HVG 2000.")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, inplace=True)
        adata = adata[:, adata.var['highly_variable']].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    metadata_rows = []
    valid_barcodes = set(adata.obs.index)

    for idx, (barcode, x, y) in enumerate(rows):
        if (idx + 1) % 512 == 0: print(f"Processing spot {idx+1}/{len(rows)}...")
        if barcode not in valid_barcodes: continue

        patch = read_patch_from_array(wsi_array, x, y, radius)
        if patch is None: continue
        
        patch.save(os.path.join(out_image_dir, f"{barcode}.jpg"), quality=95)

        g_vec = adata[barcode, :].X
        if hasattr(g_vec, "toarray"): g_vec = g_vec.toarray()
        gene_vec = np.array(g_vec, dtype=np.float32).flatten()
        np.save(os.path.join(out_genes_dir, f"{barcode}.npy"), gene_vec)

        metadata_rows.append({
            "spot_id": barcode,
            "image_path": f"image/{barcode}.jpg",
            "gene_vector_path": f"genes/{barcode}.npy"
        })

    pd.DataFrame(metadata_rows).to_csv(os.path.join(test_data_dir, "metadata.csv"), index=False)
    print(f"Done. Saved: {len(metadata_rows)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--wsi", default=None)
    p.add_argument("--radius", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--gene_list", default=None, help="CSV with 'Name' column for gene ordering")
    args = p.parse_args()
    
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out) if args.out else root
    wsi = os.path.abspath(args.wsi) if args.wsi else autodetect_wsi(root)
    rad = args.radius if args.radius else infer_radius_from_scalefactors(root)
    produce_test_data(root, wsi, rad, out, args.gene_list)
#!/usr/bin/env python
"""VisiumHD *super-bin* preprocessing for the DiSCoTME HD model.

Groups a GxG block (default 2x2) of adjacent 16um bins into one *super-bin* and
writes a folder that the existing DiSCoTME dataset/eval/viz can consume
unchanged -- the only difference vs the regular preprocessing is that each
gene .npy is [N_sub, n_genes] (the stacked sub-bin vectors, fixed slot order)
instead of [n_genes], and the image is one patch covering the GxG footprint.

Outputs (under --out):
  image/<sbid>.jpg                 one contiguous H&E patch over the GxG region
  genes/<sbid>.npy                 [N_sub, n_genes] sub-bin HVG vectors; empty slots = 0
  metadata.csv                     spot_id, image_path, gene_vector_path
  spatial/tissue_positions.csv     spot_id, in_tissue, srow, scol, pxl_row(centroid), pxl_col(centroid)
  subbin_map.csv                   spot_id, slot, sub_barcode, sub_array_row, sub_array_col
  preprocess_subbin_report.json

Slot order for a 2x2 block (G=2):  slot = (array_row % G) * G + (array_col % G)
  -> 0=TL, 1=TR, 2=BL, 3=BR   (row increases downward, col rightward)

HVG selection / normalization mirrors preprocess_visiumhd_full.py so a super-bin
gene vector is the same HVG space, just kept per sub-bin instead of averaged.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import zarr
from PIL import Image


# ----------------------------------------------------------------------
# Helpers (mirrored from preprocess_visiumhd_full.py; kept self-contained)
# ----------------------------------------------------------------------
def read_scalefactors(root_dir: Path) -> dict:
    for p in (root_dir / "spatial" / "scalefactors_json.json", root_dir / "scalefactors_json.json"):
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing scalefactors_json.json in {root_dir}")


def resolve_square_root(root: Path) -> Path:
    if root.name == "square_016um":
        return root
    if (root / "square_016um").exists():
        return root / "square_016um"
    if (root / "binned_outputs" / "square_016um").exists():
        return root / "binned_outputs" / "square_016um"
    if (root / "filtered_feature_bc_matrix.h5").exists() and (root / "spatial").is_dir():
        return root
    raise FileNotFoundError(f"Could not locate square_016um under {root}")


def autodetect_wsi(root: Path, square_root: Path) -> Path:
    hits = []
    for search_root in (root, square_root):
        for pat in ("*.btf", "*.tif", "*.tiff"):
            hits.extend(glob.glob(str(search_root / pat)))
    if hits:
        return Path(sorted(hits)[0]).resolve()
    raise FileNotFoundError(f"No .btf/.tif/.tiff WSI found under {root} or {square_root}")


def open_wsi_zarr(wsi_path: Path):
    """Lazy WSI handle. Handles pyramidal (multiscale) TIFFs by using level 0."""
    print(f"Opening WSI lazily: {wsi_path}", flush=True)
    store = tifffile.imread(str(wsi_path), aszarr=True)
    arr = zarr.open(store, mode="r")
    if not hasattr(arr, "shape"):  # multiscale -> zarr Group; take full-res level 0
        keys = sorted(arr.array_keys(), key=int)
        print(f"Multiscale WSI: levels {keys}; using full-res level {keys[0]}", flush=True)
        arr = arr[keys[0]]
    return store, arr


def extract_patch(wsi_arr, x: int, y: int, radius: int):
    y0, y1 = y - radius, y + radius
    x0, x1 = x - radius, x + radius
    if y0 < 0 or x0 < 0 or y1 > wsi_arr.shape[0] or x1 > wsi_arr.shape[1]:
        return None
    return wsi_arr[y0:y1, x0:x1]


def normalize_rgb(arr) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else a * 0
        arr = np.nan_to_num(a, nan=0, posinf=255, neginf=0).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def preprocess_genes_like_regular(square_root: Path, hvg_method: str, n_genes: int):
    h5_file = square_root / "filtered_feature_bc_matrix.h5"
    if not h5_file.exists():
        raise FileNotFoundError(f"Missing filtered_feature_bc_matrix.h5: {h5_file}")
    print(f"Loading expression matrix: {h5_file}", flush=True)
    try:
        adata = sc.read_10x_h5(str(h5_file))
    except ValueError:
        adata = sc.read_h5ad(str(h5_file))
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata = adata[:, ~adata.var["mt"]].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if hvg_method == "seurat_v3":
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_genes, layer="counts", inplace=True)
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, inplace=True)
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"Final data shape: {adata.shape}", flush=True)
    return adata


def infer_radius(square_root: Path, grid: int) -> int:
    """Default patch radius covers the GxG super-bin footprint."""
    sf = read_scalefactors(square_root)
    sub_radius = max(int(math.floor(float(sf["spot_diameter_fullres"]) / 2.0)), 1)
    return sub_radius * grid  # GxG block -> ~G*diameter patch side


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def produce(args):
    root = Path(args.root).expanduser().resolve()
    square_root = resolve_square_root(root)
    out_dir = Path(args.out).expanduser().resolve()
    (out_dir / "image").mkdir(parents=True, exist_ok=True)
    (out_dir / "genes").mkdir(parents=True, exist_ok=True)
    (out_dir / "spatial").mkdir(parents=True, exist_ok=True)

    G = args.grid
    N_sub = G * G
    wsi_path = Path(args.wsi).expanduser().resolve() if args.wsi else autodetect_wsi(root, square_root)
    radius = args.radius if args.radius is not None else infer_radius(square_root, G)
    patch_size = 2 * radius
    print(f"Grid={G}x{G} (N_sub={N_sub}); image patch radius={radius} -> {patch_size}x{patch_size}px", flush=True)

    # --- genes (HVG space, same as regular) ---
    adata = preprocess_genes_like_regular(square_root, args.hvg_method, args.n_genes)
    n_genes = adata.shape[1]
    bc_to_idx = {bc: i for i, bc in enumerate(adata.obs_names)}
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X.astype(np.float32)

    # --- positions (16um bins) ---
    pos_p = square_root / "spatial" / "tissue_positions.parquet"
    if not pos_p.exists():
        raise FileNotFoundError(f"Missing tissue_positions.parquet: {pos_p}")
    df = pd.read_parquet(pos_p)
    df = df[df["in_tissue"] == 1].copy()
    df["barcode"] = df["barcode"].astype(str)
    df = df[df["barcode"].isin(bc_to_idx)].copy()

    # optional image-space quadrant (same convention as preprocess_visiumhd_full.py)
    if args.quadrant:
        rmid = (df["pxl_row_in_fullres"].min() + df["pxl_row_in_fullres"].max()) / 2.0
        cmid = (df["pxl_col_in_fullres"].min() + df["pxl_col_in_fullres"].max()) / 2.0
        bottom = df["pxl_row_in_fullres"] >= rmid
        right = df["pxl_col_in_fullres"] >= cmid
        masks = {"br": bottom & right, "bl": bottom & ~right, "tr": ~bottom & right, "tl": ~bottom & ~right}
        if args.quadrant not in masks:
            raise ValueError(f"Unknown quadrant '{args.quadrant}'")
        df = df[masks[args.quadrant]].copy()
        print(f"Quadrant '{args.quadrant}': kept {len(df)} in-tissue bins", flush=True)

    # --- group bins into super-bins ---
    df["srow"] = df["array_row"] // G
    df["scol"] = df["array_col"] // G
    df["slot"] = (df["array_row"] % G) * G + (df["array_col"] % G)
    if args.max_superbins is not None:
        keep = df[["srow", "scol"]].drop_duplicates().head(args.max_superbins)
        df = df.merge(keep, on=["srow", "scol"], how="inner")

    store, wsi_arr = open_wsi_zarr(wsi_path)
    H, W = wsi_arr.shape[0], wsi_arr.shape[1]

    metadata, positions, submap, skipped = [], [], [], []
    groups = df.groupby(["srow", "scol"], sort=True)
    total = groups.ngroups
    print(f"Total super-bins: {total}", flush=True)

    try:
        for i, ((srow, scol), g) in enumerate(groups, start=1):
            sbid = f"sb_{int(srow):05d}_{int(scol):05d}"
            cy = int(round(g["pxl_row_in_fullres"].astype(float).mean()))
            cx = int(round(g["pxl_col_in_fullres"].astype(float).mean()))

            image_rel = f"image/{sbid}.jpg"
            gene_rel = f"genes/{sbid}.npy"
            ip = out_dir / image_rel
            gp = out_dir / gene_rel
            if args.resume and ip.exists() and gp.exists():
                metadata.append({"spot_id": sbid, "image_path": image_rel, "gene_vector_path": gene_rel})
                positions.append((sbid, 1, int(srow), int(scol), cy, cx))
                continue

            patch = extract_patch(wsi_arr, cx, cy, radius)
            if patch is None or patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                skipped.append({"spot_id": sbid, "x": cx, "y": cy, "reason": "patch out of bounds"})
                continue

            # stacked sub-bin gene vectors [N_sub, n_genes], empty slots = 0
            sub = np.zeros((N_sub, n_genes), dtype=np.float32)
            for _, r in g.iterrows():
                slot = int(r["slot"])
                sub[slot] = X[bc_to_idx[r["barcode"]]]
                submap.append({"spot_id": sbid, "slot": slot, "sub_barcode": r["barcode"],
                               "sub_array_row": int(r["array_row"]), "sub_array_col": int(r["array_col"])})

            normalize_rgb(patch).save(ip, quality=args.jpeg_quality)
            np.save(gp, sub)
            metadata.append({"spot_id": sbid, "image_path": image_rel, "gene_vector_path": gene_rel})
            positions.append((sbid, 1, int(srow), int(scol), cy, cx))

            if i % args.progress_every == 0:
                print(f"Processed {i}/{total} super-bins; written={len(metadata)} skipped={len(skipped)}", flush=True)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    pd.DataFrame(metadata).to_csv(out_dir / "metadata.csv", index=False)
    # tissue_positions.csv: same 6-col, header-less format the dataset expects
    pd.DataFrame(positions).to_csv(out_dir / "spatial" / "tissue_positions.csv", index=False, header=False)
    pd.DataFrame(submap).to_csv(out_dir / "subbin_map.csv", index=False)
    with open(out_dir / "preprocess_subbin_report.json", "w") as f:
        json.dump({"root": str(root), "square_root": str(square_root), "wsi": str(wsi_path),
                   "grid": G, "n_sub": N_sub, "radius": radius, "patch_size": patch_size,
                   "n_genes": int(n_genes), "hvg_method": args.hvg_method, "quadrant": args.quadrant,
                   "written": len(metadata), "skipped": len(skipped)}, f, indent=2)
    print(f"Done. {len(metadata)} super-bins -> {out_dir} (skipped {len(skipped)})", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="sample folder / outs / square_016um")
    p.add_argument("--wsi", default=None, help="full-res WSI (.tif/.btf); autodetected if omitted")
    p.add_argument("--out", required=True)
    p.add_argument("--grid", type=int, default=2, help="GxG sub-bins per super-bin (2 = 2x2)")
    p.add_argument("--radius", type=int, default=None, help="image patch radius px (default covers GxG)")
    p.add_argument("--hvg_method", "--hvg-method", choices=["default", "seurat_v3"], default="default")
    p.add_argument("--n_genes", "--n-genes", type=int, default=2000)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--max-superbins", type=int, default=None, help="cap for debugging")
    p.add_argument("--quadrant", choices=["br", "bl", "tr", "tl"], default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    produce(parse_args())

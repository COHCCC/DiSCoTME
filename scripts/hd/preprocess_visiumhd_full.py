#!/usr/bin/env python
"""Full Visium HD preprocessing with regular-Visium gene logic.

Outputs:
  image/<barcode>.jpg
  genes/<barcode>.npy
  metadata.csv
  spatial/tissue_positions.csv

This avoids constructing full SpatialData shapes for all bins. Coordinates come
from tissue_positions.parquet and image patches are read lazily from the full
resolution BigTIFF/BTIFF.
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


def read_scalefactors(root_dir: Path) -> dict:
    for p in (root_dir / "spatial" / "scalefactors_json.json", root_dir / "scalefactors_json.json"):
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing scalefactors_json.json in {root_dir}")


def infer_radius_from_scalefactors(root_dir: Path) -> int:
    sf = read_scalefactors(root_dir)
    return max(int(math.floor(float(sf["spot_diameter_fullres"]) / 2.0)), 1)


def resolve_square_root(root: Path) -> Path:
    if root.name == "square_016um":
        return root
    if (root / "square_016um").exists():
        return root / "square_016um"
    if (root / "binned_outputs" / "square_016um").exists():
        return root / "binned_outputs" / "square_016um"
    # Flat layout: h5 + spatial/ live directly under root (e.g. CRC_Moonshot samples).
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


def load_positions(square_root: Path, adata_obs_names, quadrant: str = None) -> pd.DataFrame:
    p = square_root / "spatial" / "tissue_positions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing tissue_positions.parquet: {p}")
    df = pd.read_parquet(p)
    df = df[df["in_tissue"] == 1].copy()
    if quadrant:
        # Quadrants are defined in IMAGE/PIXEL space (row increases downward,
        # col increases rightward), split at the midpoint of the in-tissue bbox.
        rmid = (df["pxl_row_in_fullres"].min() + df["pxl_row_in_fullres"].max()) / 2.0
        cmid = (df["pxl_col_in_fullres"].min() + df["pxl_col_in_fullres"].max()) / 2.0
        bottom = df["pxl_row_in_fullres"] >= rmid
        right = df["pxl_col_in_fullres"] >= cmid
        masks = {"br": bottom & right, "bl": bottom & ~right,
                 "tr": ~bottom & right, "tl": ~bottom & ~right}
        if quadrant not in masks:
            raise ValueError(f"Unknown quadrant '{quadrant}' (choose br/bl/tr/tl)")
        n_before = len(df)
        df = df[masks[quadrant]].copy()
        print(f"Quadrant '{quadrant}' (pixel space): kept {len(df)}/{n_before} in-tissue bins "
              f"(rmid={rmid:.0f}, cmid={cmid:.0f})", flush=True)
    df["barcode"] = df["barcode"].astype(str)
    df = df[df["barcode"].isin(adata_obs_names)].copy()
    if df.empty:
        raise ValueError("No in-tissue positions overlap filtered AnnData barcodes.")
    return df


def write_discotme_spatial_csv(positions: pd.DataFrame, out_dir: Path) -> Path:
    spatial_dir = out_dir / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    out_path = spatial_dir / "tissue_positions.csv"
    cols = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]
    positions.loc[:, cols].to_csv(out_path, index=False, header=False)
    return out_path


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
        print(f"--> Using seurat_v3 (from layers['counts']) to find {n_genes} HVGs", flush=True)
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=n_genes,
            layer="counts",
            inplace=True,
        )
    else:
        print(f"--> Using default method (from log-normalized adata.X) to find {n_genes} HVGs", flush=True)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, inplace=True)

    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"Final data shape: {adata.shape}", flush=True)
    return adata


def open_wsi_zarr(wsi_path: Path):
    print(f"Opening WSI lazily: {wsi_path}", flush=True)
    store = tifffile.imread(str(wsi_path), aszarr=True)
    arr = zarr.open(store, mode="r")
    # Pyramidal/multiscale TIFFs (e.g. ndpi2tiff output) open as a zarr Group
    # with one array per level ("0".."N"); pick the full-resolution level 0.
    # Single-level TIFFs open directly as an Array and are returned as-is.
    if not hasattr(arr, "shape"):
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


def save_gene_vector(adata, barcode: str, out_path: Path) -> int:
    vec = adata[barcode, :].X
    if hasattr(vec, "toarray"):
        vec = vec.toarray()
    elif hasattr(vec, "A"):
        vec = vec.A
    vec = np.asarray(vec, dtype=np.float32).flatten()
    np.save(out_path, vec)
    return int(vec.size)


def produce_full_data(args):
    root = Path(args.root).expanduser().resolve()
    square_root = resolve_square_root(root)
    out_dir = Path(args.out).expanduser().resolve()
    image_dir = out_dir / "image"
    genes_dir = out_dir / "genes"
    image_dir.mkdir(parents=True, exist_ok=True)
    genes_dir.mkdir(parents=True, exist_ok=True)

    wsi_path = Path(args.wsi).expanduser().resolve() if args.wsi else autodetect_wsi(root, square_root)
    radius = args.radius if args.radius is not None else infer_radius_from_scalefactors(square_root)
    patch_size = 2 * radius
    print(f"Using radius={radius}; expected patch size={patch_size}x{patch_size}", flush=True)

    adata = preprocess_genes_like_regular(square_root, args.hvg_method, args.n_genes)
    positions = load_positions(square_root, adata.obs_names, quadrant=args.quadrant)
    if args.max_spots is not None:
        positions = positions.head(args.max_spots).copy()
    spatial_csv = write_discotme_spatial_csv(positions, out_dir)
    print(f"Wrote DiSCoTME spatial coordinates: {spatial_csv}", flush=True)
    print(f"Writing {len(positions)} spots to {out_dir}", flush=True)

    store, wsi_arr = open_wsi_zarr(wsi_path)
    metadata = []
    skipped = []

    try:
        for i, row in enumerate(positions.itertuples(index=False), start=1):
            barcode = row.barcode
            image_rel = f"image/{barcode}.jpg"
            gene_rel = f"genes/{barcode}.npy"
            image_path = out_dir / image_rel
            gene_path = out_dir / gene_rel

            if args.resume and image_path.exists() and gene_path.exists():
                metadata.append({"spot_id": barcode, "image_path": image_rel, "gene_vector_path": gene_rel})
                continue

            x = int(float(row.pxl_col_in_fullres))
            y = int(float(row.pxl_row_in_fullres))
            patch = extract_patch(wsi_arr, x, y, radius)
            if patch is None or patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                skipped.append({"spot_id": barcode, "x": x, "y": y, "reason": "patch out of bounds"})
                continue

            normalize_rgb(patch).save(image_path, quality=args.jpeg_quality)
            save_gene_vector(adata, barcode, gene_path)
            metadata.append({"spot_id": barcode, "image_path": image_rel, "gene_vector_path": gene_rel})

            if i % args.progress_every == 0:
                pd.DataFrame(metadata).to_csv(out_dir / "metadata.partial.csv", index=False)
                print(f"Processed {i}/{len(positions)} spots; written={len(metadata)} skipped={len(skipped)}", flush=True)
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    pd.DataFrame(metadata).to_csv(out_dir / "metadata.csv", index=False)
    with open(out_dir / "preprocess_report.json", "w") as f:
        json.dump(
            {
                "root": str(root),
                "square_root": str(square_root),
                "wsi": str(wsi_path),
                "radius": radius,
                "patch_size": patch_size,
                "hvg_method": args.hvg_method,
                "n_genes": args.n_genes,
                "adata_shape": list(adata.shape),
                "written": len(metadata),
                "skipped": skipped,
            },
            f,
            indent=2,
        )
    print(f"Done. metadata.csv written to {out_dir / 'metadata.csv'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Sample folder, outs folder, or square_016um folder")
    p.add_argument("--wsi", default=None)
    p.add_argument("--radius", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--hvg_method", "--hvg-method", choices=["default", "seurat_v3"], default="default")
    p.add_argument("--n_genes", "--n-genes", type=int, default=2000)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--max-spots", type=int, default=None, help="Optional cap for debugging")
    p.add_argument("--quadrant", choices=["br", "bl", "tr", "tl"], default=None,
                   help="Keep only one image-space quadrant (br=bottom-right). Default: whole sample.")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    produce_full_data(parse_args())

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


# ---------- 工具函数 ----------

def read_scalefactors(root_dir: str) -> dict:
    """
    读取 spatial/scalefactors_json.json 并返回 dict。
    """
    cands = [
        os.path.join(root_dir, "spatial", "scalefactors_json.json"),
        os.path.join(root_dir, "scalefactors_json.json"),
    ]
    for p in cands:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"Cannot find scalefactors_json.json under {root_dir}/spatial/.")

def infer_radius_from_scalefactors(root_dir: str) -> int:
    """
    radius = floor(spot_diameter_fullres / 2)
    """
    sf = read_scalefactors(root_dir)
    if "spot_diameter_fullres" not in sf:
        raise KeyError("`spot_diameter_fullres` not found in scalefactors_json.json")
    return max(int(math.floor(float(sf["spot_diameter_fullres"]) / 2.0)), 1)

def find_tissue_csv(root_dir: str) -> str:
    """
    兼容新旧命名：tissue_positions.csv 或 tissue_positions_list.csv。
    """
    cands = [
        os.path.join(root_dir, "spatial", "tissue_positions.csv"),
        os.path.join(root_dir, "spatial", "tissue_positions_list.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find tissue_positions(.csv|_list.csv) under spatial/.")

def get_wsi_from_folder(root_dir):
    """
    依据文件夹名推断 WSI：
    root_dir = /.../CRC_08_Tumor
    期望 WSI = /.../CRC_08_Tumor/CRC_08_Tumor.(tif|tiff)
    """
    base = os.path.basename(os.path.normpath(root_dir))
    for ext in (".tif", ".tiff"):
        p = os.path.join(root_dir, base + ext)
        if os.path.exists(p):
            return p
    return None

def autodetect_wsi(root_dir):
    """
    自动查找 .tif/.tiff：优先“文件夹同名”，否则扫描当前目录。
    （不再依赖 microscope_images/）
    """
    p = get_wsi_from_folder(root_dir)
    if p is not None:
        return p
    hits = sorted(glob.glob(os.path.join(root_dir, "*.tif"))) + \
           sorted(glob.glob(os.path.join(root_dir, "*.tiff")))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot find any WSI (.tif/.tiff) in {root_dir}. Please provide --wsi.")

def load_positions(spatial_csv: str):
    """
    读取 positions（兼容是否有表头）。返回 [(barcode, x, y)] 仅保留 in_tissue=1。
    与你原始脚本语义一致：使用 full-res 像素坐标 (x, y)。
    """
    rows = []
    with open(spatial_csv, 'r') as f:
        first = f.readline().strip()
        has_header = "barcode" in first or "in_tissue" in first or "row" in first
        if not has_header and first:
            fields = first.split(',')
            if len(fields) >= 6:
                barcode = fields[0]
                in_tissue = float(fields[1])
                y = int(float(fields[4])); x = int(float(fields[5]))
                if in_tissue == 1:
                    rows.append((barcode, x, y))
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(',')
            barcode = fields[0]
            in_tissue = float(fields[1])
            y = int(float(fields[4])); x = int(float(fields[5]))
            if in_tissue == 1:
                rows.append((barcode, x, y))
    return rows

def print_tiff_info(tif_path: str):
    """
    （可选）打印 TIFF 基础信息，确认维度与数据类型。
    """
    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0]
        print(f"[tifffile] series0 shape: {series.shape}  ndim={len(series.shape)}")
        try:
            print(f"[tifffile] dtype: {series.dtype}")  # 某些版本 series 可能无 dtype
        except Exception:
            pass
        # 有些 TIFF 会包含描述信息：
        try:
            desc = tif.pages[0].description
            if desc:
                print("..")
        except Exception:
            pass

def _normalize_rgb(arr):
    """
    arr: 2D (H,W) or 3D with channel last/first.
    Return: uint8 RGB PIL.Image
    """
    if arr is None:
        return None
    arr = np.asarray(arr)

    # 带 alpha 的去掉 alpha
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    # 通道在前的情况 (C,H,W) -> (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        # 灰度转 RGB
        if arr.dtype == np.uint16:
            img = Image.fromarray(arr, mode="I;16").convert("RGB")
        else:
            # 其他类型一律先归一到 0-255 再转 uint8
            if arr.dtype != np.uint8:
                a = arr.astype(np.float32)
                lo, hi = float(a.min()), float(a.max())
                if hi > lo:
                    a = (a - lo) / (hi - lo) * 255.0
                else:
                    a[:] = 0
                arr = a.astype(np.uint8)
            img = Image.fromarray(arr, mode="L").convert("RGB")
        return img

    if arr.ndim == 3:
        # 确保 uint8
        if arr.dtype != np.uint8:
            a = arr.astype(np.float32)
            lo, hi = float(a.min()), float(a.max())
            if hi > lo:
                a = (a - lo) / (hi - lo) * 255.0
            else:
                a[:] = 0
            arr = a.astype(np.uint8)
        img = Image.fromarray(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    raise ValueError(f"Unexpected array ndim: {arr.ndim}, shape={arr.shape}")


def read_patch_tifffile(tif_path, x_center, y_center, radius):
    """
    在原分辨率下，仅从 TIFF 中读取以 (x_center, y_center) 为中心、边长约 2*radius 的方形窗口。
    读取策略（从快到稳依次回退）：
      A) tifffile + zarr 分块切片（若可用）
      B) tifffile.memmap（仅少数未压缩/连续 TIFF 可用）
      C) OpenSlide.read_region（pyramidal/压缩/tiles 的 WSI 常用）
      D) 兜底：整图读取后再切片（最后手段，可能占用大量内存）
    返回：PIL.Image (RGB)；窗口越界或为空则返回 None。
    依赖：外部存在 _normalize_rgb(arr) -> PIL.Image 的工具函数。
    """
    # 先获知原图宽高（不把整图进内存）
    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0]
        shp = series.shape
        if len(shp) == 2:
            H, W = shp
            layout = "HW"
        elif len(shp) == 3:
            # 判断通道维在前还是在后
            if shp[0] in (3, 4) and shp[0] < shp[1] and shp[0] < shp[2]:
                # (C, H, W)
                H, W = shp[1], shp[2]
                layout = "CHW"
            else:
                # (H, W, C)
                H, W = shp[0], shp[1]
                layout = "HWC"
        else:
            H, W = shp[-2], shp[-1]
            layout = "...HW"

    # 计算窗口，并裁剪到图像边界内
    x0 = max(0, int(round(x_center - radius)))
    y0 = max(0, int(round(y_center - radius)))
    x1 = min(W, int(round(x_center + radius)))
    y1 = min(H, int(round(y_center + radius)))
    if x1 <= x0 or y1 <= y0:
        return None

    # ---------- A) tifffile + zarr 分块切片 ----------
    try:
        import zarr  # 需要安装 zarr
        with tifffile.TiffFile(tif_path) as tif:
            ser = tif.series[0]
            # 优先 series.aszarr；旧版本回退 tif.aszarr(series=0)
            za = ser.aszarr() if hasattr(ser, "aszarr") else tif.aszarr(series=0)
            # 某些版本返回的是“store”，需 zarr.open 成为 array
            if not hasattr(za, "dtype"):
                za = zarr.open(za, mode="r")

            shp = getattr(za, "shape", None)
            if shp is None:
                raise RuntimeError("zarr array has no shape")

            if len(shp) == 2:
                arr = np.asarray(za[y0:y1, x0:x1])
            elif len(shp) == 3 and shp[0] in (3, 4) and shp[0] < shp[1] and shp[0] < shp[2]:
                # (C, H, W)
                arr = np.asarray(za[:, y0:y1, x0:x1])
            else:
                # (H, W, C) 或其余通道在后的布局
                arr = np.asarray(za[y0:y1, x0:x1, ...])

            if arr.size == 0:
                return None
            return _normalize_rgb(arr)
    except Exception:
        pass  # 回退到 B

    # ---------- B) tifffile.memmap ----------
    try:
        mm = tifffile.memmap(tif_path)  # 部分压缩/tiles/pyramidal 会失败
        shp = mm.shape
        if len(shp) == 2:
            sl = mm[y0:y1, x0:x1]
        elif len(shp) == 3 and shp[0] in (3, 4) and shp[0] < shp[1] and shp[0] < shp[2]:
            # (C, H, W)
            sl = mm[:, y0:y1, x0:x1]
        else:
            # (H, W, C) 或通道在后的布局
            sl = mm[y0:y1, x0:x1, ...]
        arr = np.asarray(sl).copy()  # 拷贝，避免映射关闭后失效
        if arr.size == 0:
            return None
        return _normalize_rgb(arr)
    except Exception:
        pass  # 回退到 C

    # ---------- C) OpenSlide（更适合 WSI） ----------
    try:
        from openslide import OpenSlide
        slide = OpenSlide(tif_path)
        w, h = (x1 - x0), (y1 - y0)
        region = slide.read_region((x0, y0), 0, (w, h))  # level=0 原分辨率
        slide.close()
        arr = np.asarray(region)[..., :3]  # RGBA -> RGB
        if arr.size == 0:
            return None
        return _normalize_rgb(arr)
    except Exception:
        pass  # 回退到 D

    # ---------- D) 兜底：整图读取后再切片（最后手段） ----------
    try:
        with tifffile.TiffFile(tif_path) as tif:
            base = tif.series[0].asarray()  # 警告：可能整图进内存
        if base.ndim == 3 and base.shape[0] in (3, 4) and base.shape[0] < base.shape[-1]:
            base = np.moveaxis(base, 0, -1)  # (C,H,W) -> (H,W,C)
        arr = base[y0:y1, x0:x1]
        if arr.size == 0:
            return None
        return _normalize_rgb(arr)
    except Exception as e:
        raise RuntimeError(f"All backends failed to read patch from {tif_path}: {e}")
    
    
# def read_patch_tifffile(tif_path, x_center, y_center, radius):
#     """
#     从普通平面 TIFF 仅读取指定窗口（原分辨率，不整图进内存）。
#     优先使用 tifffile.aszarr (需 zarr)，失败则回退到 tifffile.memmap。
#     返回 PIL.Image (RGB)。越界裁剪；窗口为空返回 None。
#     """
#     # 计算窗口
#     with tifffile.TiffFile(tif_path) as tif:
#         series = tif.series[0]
#         H = series.shape[0]
#         W = series.shape[1]

#     x0 = max(0, int(round(x_center - radius)))
#     y0 = max(0, int(round(y_center - radius)))
#     x1 = min(W, int(round(x_center + radius)))
#     y1 = min(H, int(round(y_center + radius)))
#     if x1 <= x0 or y1 <= y0:
#         return None

#     # --- 路径 A：aszarr（更稳，按块读取；需要 zarr） ---
#     try:
#         import zarr  # noqa: F401
#         with tifffile.TiffFile(tif_path) as tif:
#             z = tif.aszarr(series=0)  # level=0 默认原分辨率
#             # 直接对 zarr 数组切片，只读所需窗口
#             arr = np.asarray(z[y0:y1, x0:x1])
#     except Exception:
#         # --- 路径 B：memmap（无需 zarr；对大图是惰性页式访问）---
#         mm = tifffile.memmap(tif_path)
#         # memmap 返回的是整个数组的内存映射；切片只会触发该片段的 I/O
#         arr = np.asarray(mm[y0:y1, x0:x1]).copy()  # .copy() 以免后续关闭映射影响

#     # 统一转 RGB 保存
#     if arr.ndim == 2:
#         img = Image.fromarray(arr, mode="I;16" if arr.dtype == np.uint16 else "L").convert("RGB")
#     elif arr.ndim == 3:
#         img = Image.fromarray(arr)
#         if img.mode != "RGB":
#             img = img.convert("RGB")
#     else:
#         raise ValueError(f"Unexpected TIFF window shape: {arr.shape}")
#     return img

# ---------- 核心逻辑（输出与原始硬编码版一致） ----------

def produce_test_data(
    root_dir: str,           # outs/ 目录
    wsi_filename: str,       # .tif/.tiff 路径（普通平面TIFF）
    radius: int,             # patch 半径（像素, full-res）
    test_data_dir: str       # 输出目录；通常 = root_dir（保持原版输出）
):
    print("Loading path!")
    spatial_csv = find_tissue_csv(root_dir)
    h5_file     = os.path.join(root_dir, "filtered_feature_bc_matrix.h5")
    wsi_path    = os.path.join(wsi_filename)

    print(f"WSI path being loaded: {wsi_path}")
    out_image_dir = os.path.join(test_data_dir, "image")
    out_genes_dir = os.path.join(test_data_dir, "genes")
    metadata_csv  = os.path.join(test_data_dir, "metadata.csv")

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_genes_dir, exist_ok=True)

    # === 1) 打印 TIFF 基本信息（可选） ===
    print_tiff_info(wsi_path)

    # === 2) tissue positions ===
    rows = load_positions(spatial_csv)
    print(f"Number of spots with in_tissue=1: {len(rows)}")

    # === 3) read in spatial gene expression ===
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"Missing {h5_file}")
    adata = sc.read_10x_h5(h5_file)
    adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, inplace=True)
    adata = adata[:, adata.var['highly_variable']]
    print(f"Number of genes after filtering: {adata.shape[1]}")

    # === 4) extract patch + genes + write metadata ===
    metadata_rows = []
    for (barcode, x, y) in rows:
        # 4.1) 读取 patch（原分辨率，窗口大小约 2*radius）
        region_img = read_patch_tifffile(wsi_path, x_center=x, y_center=y, radius=radius)
        if region_img is None:
            print(f"Warning: empty patch window, skip {barcode}. (x={x}, y={y})")
            continue

        # 4.2) 保存 <barcode>.jpg
        out_img_path = os.path.join(out_image_dir, f"{barcode}.jpg")
        region_img.save(out_img_path, format="JPEG", quality=95)

        # 4.3) 保存 gene 向量
        if barcode not in adata.obs.index:
            print(f"Warning: barcode {barcode} not in adata, skip.")
            continue
        gene_vec = np.array(adata[barcode, :].X).flatten()
        out_gene_path = os.path.join(out_genes_dir, f"{barcode}.npy")
        np.save(out_gene_path, gene_vec)

        # 4.4) metadata
        metadata_rows.append({
            "spot_id": barcode,
            "image_path": f"image/{barcode}.jpg",
            "gene_vector_path": f"genes/{barcode}.npy"
        })

    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(metadata_csv, index=False)
    print(f"Metadata CSV written to {metadata_csv}")
    print("Example rows:\n", df_meta.head())


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Produce test data with tifffile (same outputs as original).")
    p.add_argument("--root", required=True,
                   help="Path to the Spaceranger outs/ directory (contains spatial/, filtered_feature_bc_matrix.h5, etc.)")
    p.add_argument("--wsi", default=None,
                   help="Path to WSI (.tif/.tiff). If omitted, will try <root_basename>.tif in <root>, else scan current dir.")
    p.add_argument("--radius", type=int, default=None,
                   help="Patch radius in pixels. If omitted, computed as floor(spot_diameter_fullres/2).")
    p.add_argument("--out", default=None,
                   help="Output directory. Default: same as --root (keeps the original layout).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = os.path.abspath(args.root)
    test_data_dir = os.path.abspath(args.out) if args.out else root_dir

    # 半径：默认从 scalefactors 推断
    if args.radius is not None:
        radius = int(args.radius)
        print(f"[INFO] Using provided radius: {radius}")
    else:
        radius = infer_radius_from_scalefactors(root_dir)
        print(f"[INFO] Inferred radius from scalefactors: {radius}")

    # WSI：优先“文件夹同名”，否则扫描当前目录
    if args.wsi is not None:
        wsi_filename = os.path.abspath(args.wsi)
        print(f"[INFO] Using provided WSI: {wsi_filename}")
    else:
        wsi_filename = autodetect_wsi(root_dir)
        print(f"[INFO] Auto-detected WSI: {wsi_filename}")

    print(f"root_dir: {root_dir}")
    print(f"wsi_filename: {wsi_filename}")
    print(f"radius: {radius}")
    print(f"test_data_dir: {test_data_dir}")

    produce_test_data(root_dir, wsi_filename, radius, test_data_dir)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import shutil

def produce_test_data(
    root_dir,                
    wsi_filename,             
    radius,                   
    test_data_dir
):
    """
	1.	Extract patches from the Whole Slide Image (WSI) based on tissue_positions_list.csv, naming them as <barcode>.jpg.
	2.	Extract gene vectors from filtered_feature_bc_matrix.h5 and save them as <barcode>.npy.
	3.	Generate a metadata.csv file containing [spot_id, image_path, gene_vector_path].
    """
    print("Loading path!")
    # spatial_csv = os.path.join(root_dir, "spatial", "tissue_positions_list.csv")
    spatial_csv = os.path.join(root_dir, "spatial", "tissue_positions.csv")
    h5_file     = os.path.join(root_dir, "filtered_feature_bc_matrix.h5")
    wsi_path    = os.path.join(wsi_filename)

    print(f"WSI path being loaded: {wsi_path}")
    out_image_dir = os.path.join(test_data_dir, "image")
    out_genes_dir = os.path.join(test_data_dir, "genes")
    metadata_csv  = os.path.join(test_data_dir, "metadata.csv")

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_genes_dir, exist_ok=True)

    # === 1) Whole Slide Image
    wsi_img = cv2.imread(wsi_path)
    print(wsi_path)
    if wsi_img is None:
        raise FileNotFoundError(f"Cannot load WSI at: {wsi_path}")
    print(f"WSI loaded, shape: {wsi_img.shape}")

    # === 2) tissue position CSV (can have header or not). barcode, in_tissue, row, col, y, x
    #         eg:
    #         ACGCTGACACGCGCT-1,0,0,0,1993,1608
    #         TACCGATCAACACTT-1,0,1,1,2146,1877
    #         ...
    rows = []
    with open(spatial_csv, 'r') as f:
        # for newer verison spaceranger output (with header)
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split(',')
            # fields[0]=barcode, fields[1]=in_tissue, fields[2]=row, fields[3]=col, fields[4]=y, fields[5]=x
            barcode    = fields[0]
            in_tissue  = float(fields[1])
            row_i      = float(fields[2])  # not used here, but we keep it if needed
            col_i      = float(fields[3])
            y = int(float(fields[4]))
            x = int(float(fields[5]))
            # in_tissue  = int(fields[1])
            # row_i      = int(fields[2])  # not used here, but we keep it if needed
            # col_i      = int(fields[3])
            # y          = int(fields[4])
            # x          = int(fields[5])
            # only under tissue
            if in_tissue == 1:
                rows.append((barcode, x, y))

    print(f"Number of spots with in_tissue=1: {len(rows)}")

    # === 3) read in spatial gene expression
    adata = sc.read_10x_h5(h5_file)
    adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, inplace=True)
    adata = adata[:, adata.var['highly_variable']]
    print(f"Number of genes after filtering: {adata.shape[1]}")

    # === 4) extract patch + genes + write metadata
    metadata_rows = []

    for (barcode, x, y) in rows:
        # 4.1) patch
        patch = wsi_img[y - radius : y + radius, x - radius : x + radius]
        if patch.size == 0:
            print(f"Warning: patch out of range, skip {barcode}. (x={x}, y={y})")
            continue

        # 4.2) write <barcode>.jpg
        out_img_path = os.path.join(out_image_dir, f"{barcode}.jpg")
        cv2.imwrite(out_img_path, patch)

        # 4.3) write gene vector -> <barcode>.npy
        #      adata[barcode, :].X shape = (1, n_genes)
        if barcode not in adata.obs.index:
            print(f"Warning: barcode {barcode} not in adata, skip.")
            continue
        gene_vec = adata[barcode, :].X
        gene_vec = np.array(gene_vec).flatten()  # shape (n_genes,)
        
        out_gene_path = os.path.join(out_genes_dir, f"{barcode}.npy")
        np.save(out_gene_path, gene_vec)

        # 4.4) metadata
        #     spot_id = barcode
        #     image_path, gene_vector_path => related test_data dict
        #     "image/AAACAGAGCGACTCAG-1.jpg", "genes/AAACAGAGCGACTCAG-1.npy"
        metadata_rows.append({
            "spot_id": barcode,
            "image_path": f"image/{barcode}.jpg",
            "gene_vector_path": f"genes/{barcode}.npy"
        })

    # === 5) metadata.csv
    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(metadata_csv, index=False)
    print(f"Metadata CSV written to {metadata_csv}")
    print("Example rows:\n", df_meta.head())


if __name__ == "__main__":
    root_dir      = "/coh_labs/dits/nsong/ovarian/J18_realignment/outs"
    wsi_filename  = "/coh_labs/dits/nsong/ovarian/J18/outs/J18.tif"   
    radius        = 76
    test_data_dir = root_dir         
    print(f"root_dir: {root_dir}")
    print(f"wsi_filename: {wsi_filename}")
    produce_test_data(root_dir, wsi_filename, radius, test_data_dir)
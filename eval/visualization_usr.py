#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm import tqdm

# ========= path direction (same as training script) =========
from pathlib import Path

# Allow environment variable override
ENV_ROOT = os.environ.get("STHISTOCLIP_ROOT")

# Default: infer from script location
guess = Path(__file__).resolve()
candidates = [Path(ENV_ROOT).resolve()] if ENV_ROOT else []
candidates += [
    guess.parents[1],        # Assume script is in <repo>/TME_aware_model/
    guess.parents[2],        # Alternative: one more level up
]

added = False
for root in candidates:
    if root.is_dir() and (root / "TME_aware_model").exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        added = True
        print(f"[INFO] PROJECT_ROOT added to sys.path: {root}")
        break

if not added:
    print("[WARN] Could not locate project root automatically. "
          "Set STHISTOCLIP_ROOT=/path/to/repo to override.")

# ========= actual imports =========
from TME_aware_model.global_local_model import (
    GlobalLocalContextDataset, GlobalLocalAwareEncoder,
    ImageEncoder, ContextAwareImageEncoder, ContextAwareGeneEncoder,
    GlobalLocalMultiModalModel
)
from TME_aware_model.global_local_train import train_one_epoch, contrastive_loss


def parse_args():
    parser = argparse.ArgumentParser(description="DiSCoTME visualization and clustering")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing the data")
    parser.add_argument("--metadata-csv", type=str, required=True,
                        help="Path to metadata CSV file")
    parser.add_argument("--tissue-positions-csv", type=str, default="tissue_positions.csv",
                        help="Path to tissue positions CSV file")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--embed-dim", type=int, default=256,
                        help="Embedding dimension (must match training)")
    parser.add_argument("--proj-dim", type=int, default=128,
                        help="Projection dimension (must match training)")
    parser.add_argument("--image-backbone", type=str, default="vit_small_patch16_224_dino",
                        help="Image encoder backbone (must match training)")
    parser.add_argument("--context-config", type=str, default="LongNet_for_spatial",
                        help="Context configuration name (must match training)")
    
    # Dataset arguments (must match training)
    parser.add_argument("--num-local", type=int, default=15,
                        help="Number of local context patches")
    parser.add_argument("--num-global", type=int, default=0,
                        help="Number of global context patches")
    parser.add_argument("--local-distance", type=int, default=400,
                        help="Maximum distance for local context")
    
    # Processing arguments
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for embedding extraction")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of data loading workers")
    
    # Clustering arguments
    parser.add_argument("--k-min", type=int, default=4,
                        help="Minimum number of clusters")
    parser.add_argument("--k-max", type=int, default=10,
                        help="Maximum number of clusters")
    parser.add_argument("--n-components-pca", type=int, default=20,
                        help="Number of PCA components")
    parser.add_argument("--n-init", type=int, default=50,
                        help="Number of KMeans initializations")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--output-suffix", type=str, default="e5",
                        help="Suffix for output files (e.g., 'e5' for epoch 5)")
    
    # GPU arguments
    parser.add_argument("--gpu-ids", type=str, default="0,1",
                        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')")
    
    return parser.parse_args()


def do_pca_tsne_kmeans(emb_matrix, k=5, n_components_pca=20, n_init=50):
    """Perform PCA dimensionality reduction followed by KMeans clustering"""
    # PCA
    pca = PCA(n_components=n_components_pca, random_state=42)
    pca_emb = pca.fit_transform(emb_matrix)
    
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=n_init)
    cluster_labels = kmeans.fit_predict(pca_emb)
    
    return cluster_labels


def extract_all_embeddings(model, dataloader, device="cuda"):
    """Extract embeddings from all spots in the dataset"""
    model.eval()
    all_img = []
    all_gene = []
    all_fuse = []
    all_spot = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            img_emb, gene_emb = model(batch)
            
            all_img.append(img_emb.cpu())
            all_gene.append(gene_emb.cpu())

            fuse_emb = (img_emb + gene_emb) / 2
            all_fuse.append(fuse_emb.cpu())

            all_spot.extend(batch["target_id"])

    img_matrix = torch.cat(all_img, dim=0).numpy()
    gene_matrix = torch.cat(all_gene, dim=0).numpy()
    fuse_matrix = torch.cat(all_fuse, dim=0).numpy()
    
    df_img = pd.DataFrame(img_matrix)
    df_img["spot_id"] = all_spot
    
    df_gene = pd.DataFrame(gene_matrix)
    df_gene["spot_id"] = all_spot
    
    df_fuse = pd.DataFrame(fuse_matrix)
    df_fuse["spot_id"] = all_spot
    
    return df_img, df_gene, df_fuse


def main():
    args = parse_args()
    
    # Setup device and GPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Build dataset and dataloader
    transform_image = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = GlobalLocalContextDataset(
        metadata_csv=args.metadata_csv,
        tissue_positions_csv=args.tissue_positions_csv,
        root_dir=args.data_root,
        transform_image=transform_image,
        num_local=args.num_local,
        num_global=args.num_global,
        local_distance=args.local_distance
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Build model (architecture must match training)
    img_encoder = ContextAwareImageEncoder(
        model_name=args.image_backbone,
        pretrained=True,
        embed_dim=args.embed_dim,
        config_name=args.context_config
    )
    
    gene_encoder = ContextAwareGeneEncoder(
        gene_dim=2000,
        embed_dim=args.embed_dim,
        config_name=args.context_config
    )
    
    model = GlobalLocalMultiModalModel(
        img_encoder=img_encoder,
        gene_encoder=gene_encoder,
        proj_dim=args.proj_dim
    )
    
    # Load checkpoint
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP training)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Setup DataParallel if using multiple GPUs
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    model.eval()
    
    # Extract embeddings
    df_img, df_gene, df_fuse = extract_all_embeddings(model, dataloader, device=device)
    print("Extracted embeddings shapes:")
    print(f"Image: {df_img.shape}, Gene: {df_gene.shape}, Fused: {df_fuse.shape}")
    
    # Save embeddings
    suffix = args.output_suffix
    
    fuse_csv = os.path.join(args.output_dir, f"fused_embeddings_{suffix}.csv")
    df_fuse.to_csv(fuse_csv, index=False)
    print(f"Saved fused embeddings to: {fuse_csv}")
    
    img_csv = os.path.join(args.output_dir, f"image_embeddings_{suffix}.csv")
    df_img.to_csv(img_csv, index=False)
    print(f"Saved image embeddings to: {img_csv}")
    
    gene_csv = os.path.join(args.output_dir, f"gene_embeddings_{suffix}.csv")
    df_gene.to_csv(gene_csv, index=False)
    print(f"Saved gene embeddings to: {gene_csv}")
    
    # Prepare matrices for clustering
    fuse_matrix = df_fuse.drop("spot_id", axis=1).values
    image_matrix = df_img.drop("spot_id", axis=1).values
    gene_matrix = df_gene.drop("spot_id", axis=1).values
    
    # Perform clustering for different k values
    for k in range(args.k_min, args.k_max + 1):
        print(f"\n--- Processing clustering with k={k} ---")
        
        k_dir = os.path.join(args.output_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)
        print(f"Created directory: {k_dir}")
        
        # Fused embedding clustering
        fused_labels = do_pca_tsne_kmeans(
            emb_matrix=fuse_matrix,
            k=k,
            n_components_pca=args.n_components_pca,
            n_init=args.n_init
        )
        
        df_fused_cluster = pd.DataFrame({
            "spot_id": df_fuse["spot_id"],
            "cluster": fused_labels
        })
        
        cluster_csv = os.path.join(k_dir, f"fused_clusters_{suffix}.csv")
        df_fused_cluster.to_csv(cluster_csv, index=False)
        print(f"Saved fused clusters to: {cluster_csv}")
        
        # Image embedding clustering
        image_labels = do_pca_tsne_kmeans(
            emb_matrix=image_matrix,
            k=k,
            n_components_pca=args.n_components_pca,
            n_init=args.n_init
        )
        
        df_image_cluster = pd.DataFrame({
            "spot_id": df_img["spot_id"],
            "cluster": image_labels
        })
        
        image_cluster_csv = os.path.join(k_dir, f"image_clusters_{suffix}.csv")
        df_image_cluster.to_csv(image_cluster_csv, index=False)
        print(f"Saved image clusters to: {image_cluster_csv}")
        
        # Gene embedding clustering
        gene_labels = do_pca_tsne_kmeans(
            emb_matrix=gene_matrix,
            k=k,
            n_components_pca=args.n_components_pca,
            n_init=args.n_init
        )
        
        df_gene_cluster = pd.DataFrame({
            "spot_id": df_gene["spot_id"],
            "cluster": gene_labels
        })
        
        gene_cluster_csv = os.path.join(k_dir, f"gene_clusters_{suffix}.csv")
        df_gene_cluster.to_csv(gene_cluster_csv, index=False)
        print(f"Saved gene clusters to: {gene_cluster_csv}")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.data.dataset import MultiScaleContextDataset
    from src.models.discotme_net import MultiScaleMultiModalModel
    from src.models.dilated_blocks import DilatedConfigs 
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)
    
def do_pca_kmeans(emb_matrix, k, pca_dim=20):
    """Standard clustering workflow"""
    if emb_matrix.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        emb_pca = pca.fit_transform(emb_matrix)
    else:
        emb_pca = emb_matrix
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    return kmeans.fit_predict(emb_pca)

def parse_args():
    parser = argparse.ArgumentParser(description="DiSCoTME Teacher vs post_training Evaluation")
    
    # --- Required Arguments ---
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    # --- Data Related ---
    parser.add_argument("--metadata-csv", type=str, default="metadata.csv")
    parser.add_argument("--tissue-positions-csv", type=str, default="tissue_positions.csv")
    parser.add_argument("--num-local", type=int, default=15)
    parser.add_argument("--num-global", type=int, default=0)
    parser.add_argument("--local-distance", type=int, default=400)
    
    # --- Model Architecture ---
    parser.add_argument("--model-arch", type=str, default="standard_discotme", 
                    choices=["standard_discotme", "factorized_discotme"],
                    help="Choose model architecture")
    parser.add_argument("--image-encoder-type", type=str, default="gated_image_encoder")
    parser.add_argument("--image-backbone", type=str, default="vit_dino_v1")
    parser.add_argument("--gene-encoder-type", type=str, default="gated_gene_encoder")
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--context-config", type=str, default="LongNet_for_spatial")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128)

    # --- Clustering Parameters ---
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--output-suffix", type=str, default="e5")
    
    # --- Runtime Environment ---
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    # [FIX] Explicitly add --device parameter
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    
    return parser.parse_args()

def extract_embeddings_with_alpha(model, dataloader, device):
    model.eval()
    results = {
        "image": [],   
        "gene": [],  
        "fused": [], 
        "alpha_img": [],     
        "alpha_gene": [],    
        "spot_ids": []
    }

    print("Extracting embeddings and Alpha weights (Tuple Unpacking)...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # 1. Forward pass: returns tuple (img_emb, gene_emb, alpha_img, alpha_gene)
            outputs = model(batch)
            
            # 2. Unpack based on indexing logic in trainer.py
            img_emb = outputs[0]    # [B, Proj_Dim]
            gene_emb = outputs[1]   # [B, Proj_Dim]
            a_img = outputs[2]      # Might be None
            a_gene = outputs[3]     # Might be None

            # 3. Compute final fused embedding (joint multi-modal space)
            fused_emb = (img_emb + gene_emb) / 2.0

            # 4. Store features
            results["image"].append(img_emb.cpu().numpy())
            results["gene"].append(gene_emb.cpu().numpy())
            results["fused"].append(fused_emb.cpu().numpy())
            results["spot_ids"].extend(batch['target_id'])

            # 5. Handle Alpha (prevent flatten errors if None)
            batch_size = img_emb.size(0)
            if a_img is not None:
                results["alpha_img"].append(a_img.cpu().numpy())
            else:
                # If no alpha, fill with 1 (represents 100% local)
                results["alpha_img"].append(np.ones((batch_size, 1), dtype=np.float32))
                
            if a_gene is not None:
                results["alpha_gene"].append(a_gene.cpu().numpy())
            else:
                results["alpha_gene"].append(np.ones((batch_size, 1), dtype=np.float32))

    # Concatenate results
    final_data = {k: (np.concatenate(v, axis=0) if k != "spot_ids" else v) for k, v in results.items()}
    return final_data

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model (Ensure correct gated encoder names are passed)
    config = DilatedConfigs[args.context_config].copy() if args.context_config in DilatedConfigs else {}
    model = MultiScaleMultiModalModel(
        img_enc_name=args.image_encoder_type,
        gene_enc_name=args.gene_encoder_type,
        proj_dim=args.proj_dim,
        img_args={"backbone_type": args.image_backbone, "embed_dim": args.embed_dim, "config_dict": config},
        gene_args={"gene_dim": 2000, "embed_dim": args.embed_dim, "config_dict": config}
    )
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.to(device)

    # 2. Extract features and Alpha
    dataset = MultiScaleContextDataset(
        metadata_csv=args.metadata_csv, tissue_positions_csv=args.tissue_positions_csv,
        root_dir=args.data_root, num_local=args.num_local, num_global=args.num_global,
        transform_image=T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    data = extract_embeddings_with_alpha(model, dataloader, device)

    # 3. Clustering and saving
    suffix = args.output_suffix
    for k in range(args.k_min, args.k_max + 1):
        print(f"\n--- Processing k={k} ---")
        k_dir = os.path.join(args.output_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)

        # Clustering targets changed to fused post_training features
        for mode in ["image", "gene", "fused"]:
            labels = do_pca_kmeans(data[mode], k, args.pca_dim)
            df = pd.DataFrame({"spot_id": data["spot_ids"], "cluster": labels})
            df.to_csv(os.path.join(k_dir, f"{mode}_clusters.csv"), index=False)

    # [NEW] 4. Save Alpha weight heatmap data
    # This is the most important file for subsequent visualization
    alpha_df = pd.DataFrame({
        "spot_id": data["spot_ids"],
        "alpha_img": data["alpha_img"].flatten(),
        "alpha_gene": data["alpha_gene"].flatten()
    })
    alpha_path = os.path.join(args.output_dir, f"alpha_weights_{suffix}.csv")
    alpha_df.to_csv(alpha_path, index=False)
    print(f"\nSaved alpha weights to {alpha_path}")

    print(f"\nEvaluation complete. Results in {args.output_dir}")
    # [NEW] 5. Save raw embedding vectors (for R/Seurat analysis)
    print("\nSaving raw embeddings...")
    for mode in ["image", "gene", "fused"]:
        # Convert [N, Dim] matrix to DataFrame
        emb_df = pd.DataFrame(data[mode])
        # Insert spot_id as first column for easier matching later
        emb_df.insert(0, "spot_id", data["spot_ids"])
        
        emb_save_path = os.path.join(args.output_dir, f"{mode}_embeddings_{suffix}.csv")
        emb_df.to_csv(emb_save_path, index=False)
        print(f"Saved {mode} embeddings to {emb_save_path}")

if __name__ == "__main__":
    main()
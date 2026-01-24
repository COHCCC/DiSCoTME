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

# ==============================================================================
# [Step 1] 路径修复
# ==============================================================================
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.data.dataset import MultiScaleContextDataset
    # 引入 Registry
    from src.models.discotme_net import MODELS 
    from src.models.dilated_blocks import DilatedConfigs 
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)
    
def do_pca_kmeans(emb_matrix, k, pca_dim=20):
    """
    标准聚类流程: PCA降维 -> KMeans
    """
    # 如果特征维度太高 (比如 concat 后是 256)，先降维
    if emb_matrix.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        emb_pca = pca.fit_transform(emb_matrix)
    else:
        emb_pca = emb_matrix
        
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(emb_pca)
    return labels

def parse_args():
    parser = argparse.ArgumentParser(description="DiSCoTME Full TME Evaluation")
    
    # --- 必须参数 ---
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    # --- 模型架构 (关键) ---
    parser.add_argument("--model-arch", type=str, default="factorized_discotme", 
                        choices=["standard_discotme", "factorized_discotme"],
                        help="Choose model architecture")

    # --- 数据设置 ---
    parser.add_argument("--metadata-csv", type=str, default="metadata.csv")
    parser.add_argument("--tissue-positions-csv", type=str, default="tissue_positions.csv")
    parser.add_argument("--num-local", type=int, default=15)
    parser.add_argument("--num-global", type=int, default=0)
    parser.add_argument("--local-distance", type=int, default=400)
    
    # --- Encoder配置 ---
    parser.add_argument("--image-encoder-type", type=str, default="gated_image_encoder")
    parser.add_argument("--image-backbone", type=str, default="vit_dino_v1")
    parser.add_argument("--gene-encoder-type", type=str, default="gated_gene_encoder")
    parser.add_argument("--context-config", type=str, default="LongNet_for_spatial")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128)

    # --- 聚类参数 ---
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--output-suffix", type=str, default="eval")
    
    # --- 运行环境 ---
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()

def extract_embeddings_full(model, dataloader, device):
    """
    提取 Shared, Unique, Concat, Gene 四种特征
    """
    model.eval()
    results = {
        "z_img_shared": [],  # 生物学 (Tumor/Stroma)
        "z_img_unique": [],  # 形态学 (Hemorrhage/Necrosis)
        "z_img_concat": [],  # 全景 TME (Holistic)
        "z_gene": [],        # 基因 Ground Truth
        "alpha_img": [],     
        "alpha_gene": [],    
        "spot_ids": []
    }

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            outputs = model(batch)
            
            # --- 1. 分支处理: 获取 Shared 和 Unique ---
            if len(outputs) == 8:
                # === Factorized Model ===
                (z_shared, z_unique, z_gene, 
                 a_img, a_gene, 
                 rec_shared, rec_unique, img_raw) = outputs
                
            elif len(outputs) == 4:
                # === Standard Model (兼容旧版) ===
                z_shared, z_gene = outputs[0], outputs[1]
                a_img, a_gene = outputs[2], outputs[3]
                z_unique = torch.zeros_like(z_shared) # 占位符
                
            else:
                raise ValueError(f"Unknown output length: {len(outputs)}")

            # --- 2. 构造 Concat 特征 (Full TME) ---
            # dim=1 是特征维度: [B, 128] + [B, 128] -> [B, 256]
            z_concat = torch.cat([z_shared, z_unique], dim=1)

            # --- 3. 存储 ---
            results["z_img_shared"].append(z_shared.cpu().numpy())
            results["z_img_unique"].append(z_unique.cpu().numpy())
            results["z_img_concat"].append(z_concat.cpu().numpy())
            results["z_gene"].append(z_gene.cpu().numpy())
            results["spot_ids"].extend(batch['target_id'])

            # Alpha 处理
            batch_size = z_shared.size(0)
            if a_img is not None: results["alpha_img"].append(a_img.cpu().numpy())
            else: results["alpha_img"].append(np.ones((batch_size, 1)))
                
            if a_gene is not None: results["alpha_gene"].append(a_gene.cpu().numpy())
            else: results["alpha_gene"].append(np.ones((batch_size, 1)))

    # 拼接 numpy 数组
    final_data = {k: (np.concatenate(v, axis=0) if k != "spot_ids" else v) for k, v in results.items()}
    return final_data

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 动态加载模型
    print(f"Initializing Model: {args.model_arch}")
    ModelClass = MODELS.get(args.model_arch)
    if ModelClass is None:
         raise ValueError(f"Architecture {args.model_arch} not found in registry")
         
    config = DilatedConfigs[args.context_config].copy() if args.context_config in DilatedConfigs else {}
    
    img_args = {"backbone_type": args.image_backbone, "embed_dim": args.embed_dim, "config_dict": config}
    gene_args = {"gene_dim": 2000, "embed_dim": args.embed_dim, "config_dict": config}
    
    model = ModelClass(
        img_enc_name=args.image_encoder_type,
        gene_enc_name=args.gene_encoder_type,
        proj_dim=args.proj_dim,
        img_args=img_args,
        gene_args=gene_args
    )
    
    # 2. 加载权重
    print(f"Loading checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # 移除 DDP 前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)

    # 3. 数据加载
    dataset = MultiScaleContextDataset(
        metadata_csv=args.metadata_csv, tissue_positions_csv=args.tissue_positions_csv,
        root_dir=args.data_root, num_local=args.num_local, num_global=args.num_global,
        transform_image=T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 4. 提取特征 (全家桶)
    data = extract_embeddings_full(model, dataloader, device)

    # 5. 执行聚类 (Shared, Unique, Concat, Gene)
    suffix = args.output_suffix
    
    # 这四个是我们要做对比分析的核心对象
    targets = ["z_img_shared", "z_img_unique", "z_img_concat", "z_gene"]
    
    for k in range(args.k_min, args.k_max + 1):
        print(f"\n--- Clustering k={k} ---")
        k_dir = os.path.join(args.output_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)

        for mode in targets:
            # 如果是 Standard 模型，Unique 是全0，跳过 Unique 和 Concat 的聚类节省时间
            # 但如果你想确认它们是0也可以跑
            if mode == "z_img_unique" and np.all(data[mode] == 0):
                print(f"Skipping {mode} (all zeros detected)")
                continue
            
            # 执行聚类
            labels = do_pca_kmeans(data[mode], k, args.pca_dim)
            
            # 保存
            df = pd.DataFrame({"spot_id": data["spot_ids"], "cluster": labels})
            df.to_csv(os.path.join(k_dir, f"{mode}_clusters.csv"), index=False)
            
    print(f"\nClusters saved to {args.output_dir}")

    # 6. 保存 Alpha 权重 (画热图用)
    alpha_df = pd.DataFrame({
        "spot_id": data["spot_ids"],
        "alpha_img": data["alpha_img"].flatten(),
        "alpha_gene": data["alpha_gene"].flatten()
    })
    alpha_path = os.path.join(args.output_dir, f"alpha_weights_{suffix}.csv")
    alpha_df.to_csv(alpha_path, index=False)
    print(f"Alpha weights saved to {alpha_path}")
    
    # [NEW] 7. 顺便保存一下 z_img_concat 的 UMAP 坐标 (可选，如果你想本地直接画图)
    # 如果数据量不大 (<10万)，可以在这里算好 UMAP 存 CSV，方便画图
    # print("Computing UMAP for Concat (preview)...")
    # import umap
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    # embedding = reducer.fit_transform(data["z_img_concat"])
    # umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    # umap_df['spot_id'] = data['spot_ids']
    # umap_df.to_csv(os.path.join(args.output_dir, f"umap_concat_{suffix}.csv"), index=False)

    print("Done.")

if __name__ == "__main__":
    main()
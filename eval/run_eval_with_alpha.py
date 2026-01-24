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
# [Step 1] 路径修复 (保持你的逻辑)
# ==============================================================================
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
    """标准聚类流程"""
    if emb_matrix.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=42)
        emb_pca = pca.fit_transform(emb_matrix)
    else:
        emb_pca = emb_matrix
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    return kmeans.fit_predict(emb_pca)

def parse_args():
    parser = argparse.ArgumentParser(description="DiSCoTME Teacher vs post_training Evaluation")
    
    # --- 必须参数 ---
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    # --- 数据相关 ---
    parser.add_argument("--metadata-csv", type=str, default="metadata.csv")
    parser.add_argument("--tissue-positions-csv", type=str, default="tissue_positions.csv")
    parser.add_argument("--num-local", type=int, default=15)
    parser.add_argument("--num-global", type=int, default=0)
    parser.add_argument("--local-distance", type=int, default=400)
    
    # --- 模型架构 ---
    parser.add_argument("--model-arch", type=str, default="standard_discotme", 
                    choices=["standard_discotme", "factorized_discotme"],
                    help="Choose model architecture")
    parser.add_argument("--image-encoder-type", type=str, default="multiscale_image_encoder")
    parser.add_argument("--image-backbone", type=str, default="gigapath_with_confidence")
    parser.add_argument("--gene-encoder-type", type=str, default="gene_mlp_residual")
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--context-config", type=str, default="LongNet_for_spatial")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=128)

    # --- 聚类参数 ---
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--output-suffix", type=str, default="e5")
    
    # --- 运行环境 ---
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    # [FIX] 显式添加 --device 参数
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
            
            # 1. 前向传播：返回值为元组 (img_emb, gene_emb, alpha_img, alpha_gene)
            outputs = model(batch)
            
            # 2. 按照你 trainer.py 里的索引逻辑解包
            img_emb = outputs[0]    # [B, Proj_Dim]
            gene_emb = outputs[1]   # [B, Proj_Dim]
            a_img = outputs[2]      # 可能为 None
            a_gene = outputs[3]     # 可能为 None

            # 3. 计算最终融合嵌入 (双模态联合空间)
            fused_emb = (img_emb + gene_emb) / 2.0

            # 4. 存储特征
            results["image"].append(img_emb.cpu().numpy())
            results["gene"].append(gene_emb.cpu().numpy())
            results["fused"].append(fused_emb.cpu().numpy())
            results["spot_ids"].extend(batch['target_id'])

            # 5. 处理 Alpha (防止 None 导致 flatten 报错)
            batch_size = img_emb.size(0)
            if a_img is not None:
                results["alpha_img"].append(a_img.cpu().numpy())
            else:
                # 如果没有 alpha，填充为 1 (代表 100% 局部)
                results["alpha_img"].append(np.ones((batch_size, 1), dtype=np.float32))
                
            if a_gene is not None:
                results["alpha_gene"].append(a_gene.cpu().numpy())
            else:
                results["alpha_gene"].append(np.ones((batch_size, 1), dtype=np.float32))

    # 拼接结果
    final_data = {k: (np.concatenate(v, axis=0) if k != "spot_ids" else v) for k, v in results.items()}
    return final_data

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型 (确保传递正确的 gated 编码器名称)
    config = DilatedConfigs[args.context_config].copy() if args.context_config in DilatedConfigs else {}
    model = MultiScaleMultiModalModel(
        img_enc_name=args.image_encoder_type,
        gene_enc_name=args.gene_encoder_type,
        proj_dim=args.proj_dim,
        img_args={"backbone_type": args.image_backbone, "embed_dim": args.embed_dim, "config_dict": config},
        gene_args={"gene_dim": 2000, "embed_dim": args.embed_dim, "config_dict": config}
    )
    
    # 加载权重
    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.to(device)

    # 2. 提取特征与 Alpha
    dataset = MultiScaleContextDataset(
        metadata_csv=args.metadata_csv, tissue_positions_csv=args.tissue_positions_csv,
        root_dir=args.data_root, num_local=args.num_local, num_global=args.num_global,
        transform_image=T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    data = extract_embeddings_with_alpha(model, dataloader, device)

    # 3. 聚类并保存
    suffix = args.output_suffix
    for k in range(args.k_min, args.k_max + 1):
        print(f"\n--- Processing k={k} ---")
        k_dir = os.path.join(args.output_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)

        # 聚类对象更改为融合后的 post_training 特征
        for mode in ["image", "gene", "fused"]:
            labels = do_pca_kmeans(data[mode], k, args.pca_dim)
            df = pd.DataFrame({"spot_id": data["spot_ids"], "cluster": labels})
            df.to_csv(os.path.join(k_dir, f"{mode}_clusters.csv"), index=False)

    # [NEW] 4. 保存 Alpha 权重热图数据
    # 这是你后续做可视化最重要的文件
    alpha_df = pd.DataFrame({
        "spot_id": data["spot_ids"],
        "alpha_img": data["alpha_img"].flatten(),
        "alpha_gene": data["alpha_gene"].flatten()
    })
    alpha_path = os.path.join(args.output_dir, f"alpha_weights_{suffix}.csv")
    alpha_df.to_csv(alpha_path, index=False)
    print(f"\nSaved alpha weights to {alpha_path}")

    print(f"\nEvaluation complete. Results in {args.output_dir}")

if __name__ == "__main__":
    main()
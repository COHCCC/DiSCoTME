import os
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

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root   = os.path.abspath(os.path.join(current_dir, '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from TME_aware_model.global_local_model import GlobalLocalContextDataset, GlobalLocalAwareEncoder
from TME_aware_model.global_local_model import ImageEncoder, ContextAwareImageEncoder, ContextAwareGeneEncoder, GlobalLocalMultiModalModel
from TME_aware_model.global_local_train import train_one_epoch, contrastive_loss

# from huggingface_hub import login
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"
############################
# PCA + t-SNE + KMeans
############################
# def do_pca_tsne_kmeans(emb_matrix, 
#                        k=5, 
#                        n_components_pca=20,  #20
#                        n_init=50, 
#                        perplexity=30, 
#                        savepath=None, 
#                        title=""):

#     pca = PCA(n_components=n_components_pca, random_state=42)
#     pca_emb = pca.fit_transform(emb_matrix)
    
#     kmeans = KMeans(n_clusters=k, random_state=0, n_init=n_init)
#     cluster_labels = kmeans.fit_predict(pca_emb)
    
#     tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
#     tsne_coords = tsne.fit_transform(pca_emb)
    
#     plt.figure(figsize=(6, 6))
#     for cid in range(k):
#         mask = (cluster_labels == cid)
#         plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
#                     s=5, alpha=0.7, label=f"Cluster {cid+1}")
#     plt.title(title)
#     plt.legend()
#     if savepath:
#         plt.savefig(savepath, dpi=300)
#         plt.close()
#         print(f"[{title}] PCA->t-SNE with KMeans (k={k}) saved to {savepath}")
#     else:
#         plt.show()
    
#     return cluster_labels

def do_pca_tsne_kmeans(
    emb_matrix, 
    k=5, 
    n_components_pca=20, 
    n_init=50
):
    # PCA降维
    pca = PCA(n_components=n_components_pca, random_state=42)
    pca_emb = pca.fit_transform(emb_matrix)
    # KMeans聚类
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=n_init)
    cluster_labels = kmeans.fit_predict(pca_emb)
    return cluster_labels

def extract_all_embeddings(model, dataloader, device="cuda"):
    model.eval()
    all_img = []
    all_gene = []
    all_fuse = []
    all_spot = []

    with torch.no_grad():
        # for batch in dataloader:
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

#############################
# Laod model
#############################
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     root_dir = "/coh_labs/dits/nsong/ovarian/outs"
#     tissue_position_csv = os.path.join(root_dir, "spatial/tissue_positions.csv")
#     metadata_csv = os.path.join(root_dir, "metadata.csv")
#     # tissue_position_csv = os.path.join("/coh_labs/dits/nsong/Mayo_VisiumHD/101/outs", "binned_outputs", "square_016um", "spatial", "tissue_positions.parquet")
#     model_ckpt   = "sthistoclip_model_j13_ori.pth" 
#     batch_size   = 128
#     k_clusters   = 5
#     perplexity   = 30
#     out_dir = "/coh_labs/dits/nsong/ovarian/outs/results_ori"
#     os.makedirs(out_dir, exist_ok=True)
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    root_dir = "/coh_labs/dits/Craig_Spatial/Craig_SPA6_A/outs"
    metadata_csv = "/coh_labs/dits/Craig_Spatial/Craig_SPA6_A/outs/metadata.csv"
    tissue_positions_csv = "tissue_positions.csv"  # or .parquet for HD
    # tissue_positions_csv = os.path.join("/coh_labs/dits/nsong/Mayo_VisiumHD/101/outs", "binned_outputs", "square_016um", "spatial", "tissue_positions.parquet")
    batch_size = 128
    
    out_dir = "/coh_labs/dits/Craig_Spatial/Craig_SPA6_A/outs/results/geneattn(transformer)"
    print("Select your outdir")
    # out_dir = "/coh_labs/dits/nsong/Mayo_VisiumHD/101/outs/results/ddp"
    os.makedirs(out_dir, exist_ok=True)

    # B. Build ST dataset和 DataLoader
    transform_image = T.Compose([
        # T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = GlobalLocalContextDataset(
        metadata_csv=metadata_csv,
        tissue_positions_csv=tissue_positions_csv,
        root_dir=root_dir,
        transform_image=transform_image,  
        num_local=15,
        num_global=0,
        local_distance=400  
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
    
    # C. Load module (setting should be the same as training)
    img_encoder = ContextAwareImageEncoder(
        model_name="vit_small_patch16_224_dino",
        pretrained=True,
        embed_dim=256,
        config_name="LongNet_for_spatial"
    )
    gene_encoder = ContextAwareGeneEncoder(
        gene_dim=2000,  
        embed_dim=256,
        config_name="LongNet_for_spatial"
    )
    model = GlobalLocalMultiModalModel(
        img_encoder=img_encoder,
        gene_encoder=gene_encoder,
        proj_dim=128
    )
    
    print("Load your best model path!")
    ckpt_path = "/coh_labs/dits/nsong/python2024/DiSCoTME/TME_aware_model/checkpoints/SPA6/geneattn_ddp_20250902_131630/final_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.to(device)

    model.eval()

    df_img, df_gene, df_fuse = extract_all_embeddings(model, dataloader, device=device)
    print("Extracted embeddings shapes:")
    print(f"Image: {df_img.shape}, Gene: {df_gene.shape}, Fused: {df_fuse.shape}")

    fuse_csv = os.path.join(out_dir, "fused_embeddings_e5.csv")
    df_fuse.to_csv(fuse_csv, index=False)
    print(f"Saved fused embeddings CSV to {fuse_csv}")

    img_csv = os.path.join(out_dir, "image_embeddings_e5.csv")
    df_img.to_csv(img_csv, index=False)
    print(f"Saved image embeddings CSV to {img_csv}")

    gene_csv = os.path.join(out_dir, "gene_embeddings_e5.csv")
    df_gene.to_csv(gene_csv, index=False)
    print(f"Saved gene embeddings CSV to {gene_csv}")

    fuse_matrix = df_fuse.drop("spot_id", axis=1).values
    image_matrix = df_img.drop("spot_id", axis=1).values
    gene_matrix = df_gene.drop("spot_id", axis=1).values

    for k in range(4, 11):

        print(f"\n--- Processing clustering with k={k} ---")

        k_dir = os.path.join(out_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)
        print(f"Created directory: {k_dir}")

        # Fused embedding clustering

        fused_labels = do_pca_tsne_kmeans(
            emb_matrix=fuse_matrix,
            k=k
        )

        df_fused_cluster = pd.DataFrame({
            "spot_id": df_fuse["spot_id"],
            "cluster": fused_labels
        })

        cluster_csv = os.path.join(k_dir, "fused_clusters_e5.csv")
        df_fused_cluster.to_csv(cluster_csv, index=False)
        print(f"Saved fused clusters CSV to {cluster_csv}")

        # Image embedding clustering

        image_labels = do_pca_tsne_kmeans(
            emb_matrix=image_matrix,
            k=k
        )

        df_image_cluster = pd.DataFrame({
            "spot_id": df_img["spot_id"],
            "cluster": image_labels
        })
        image_cluster_csv = os.path.join(k_dir, "image_clusters_e5.csv")
        df_image_cluster.to_csv(image_cluster_csv, index=False)
        print(f"Saved image clusters CSV to {image_cluster_csv}")

        # Gene embedding clustering
        gene_labels = do_pca_tsne_kmeans(
            emb_matrix=gene_matrix,
            k=k
        )

        df_gene_cluster = pd.DataFrame({
            "spot_id": df_gene["spot_id"],
            "cluster": gene_labels
        })

        gene_cluster_csv = os.path.join(k_dir, "gene_clusters_e5.csv")
        df_gene_cluster.to_csv(gene_cluster_csv, index=False)

        print(f"Saved gene clusters CSV to {gene_cluster_csv}")

if __name__ == "__main__":
    main()


#     df_fuse = extract_spot_embeddings(model, dataloader, device=device)
#     print("Extracted fused embedding shape:", df_fuse.shape)
    
#     fuse_csv = os.path.join(out_dir, "fused_embeddings_e5.csv")
#     df_fuse.to_csv(fuse_csv, index=False)
#     print(f"Saved fused embeddings CSV to {fuse_csv}")
    
#     fuse_matrix = df_fuse.drop("spot_id", axis=1).values
#     fused_labels = do_pca_tsne_kmeans(
#         emb_matrix=fuse_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath=os.path.join(out_dir, "fused_kmeans_tsne_e5.png"),
#         title="Fused Embedding (Gene+Image) with KMeans"
#     )
    
#     df_fused_cluster = pd.DataFrame({
#         "spot_id": df_fuse["spot_id"],
#         "cluster": fused_labels
#     })
#     cluster_csv = os.path.join(out_dir, "fused_clusters_e5.csv")
#     df_fused_cluster.to_csv(cluster_csv, index=False)
#     print(f"Saved fused clusters CSV to {cluster_csv}")

#     df_img, df_gene = extract_image_gene_embeddings(model, dataloader, device=device)
    
#     img_csv = os.path.join(out_dir, "image_embeddings_e5.csv")
#     df_img.to_csv(img_csv, index=False)
#     print(f"Saved image embeddings CSV to {img_csv}")
    
#     gene_csv = os.path.join(out_dir, "gene_embeddings_e5.csv")
#     df_gene.to_csv(gene_csv, index=False)
#     print(f"Saved gene embeddings CSV to {gene_csv}")
    
#     image_matrix = df_img.drop("spot_id", axis=1).values
#     image_labels = do_pca_tsne_kmeans(
#         emb_matrix=image_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath=os.path.join(out_dir, "image_kmeans_tsne_e5.png"),
#         title="Image Embedding with KMeans"
#     )
#     df_image_cluster = pd.DataFrame({
#         "spot_id": df_img["spot_id"],
#         "cluster": image_labels
#     })
#     image_cluster_csv = os.path.join(out_dir, "image_clusters_e5.csv")
#     df_image_cluster.to_csv(image_cluster_csv, index=False)
#     print(f"Saved image clusters CSV to {image_cluster_csv}")
    
#     # 可视化纯 gene embedding
#     gene_matrix = df_gene.drop("spot_id", axis=1).values
#     gene_labels = do_pca_tsne_kmeans(
#         emb_matrix=gene_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath=os.path.join(out_dir, "gene_kmeans_tsne_e5.png"),
#         title="Gene Embedding with KMeans"
#     )
#     df_gene_cluster = pd.DataFrame({
#         "spot_id": df_gene["spot_id"],
#         "cluster": gene_labels
#     })
#     gene_cluster_csv = os.path.join(out_dir, "gene_clusters_e5.csv")
#     df_gene_cluster.to_csv(gene_cluster_csv, index=False)
#     print(f"Saved gene clusters CSV to {gene_cluster_csv}")
# if __name__ == "__main__":
#     main()


# def do_tsne_kmeans(emb_matrix, 
#                    k=5, 
#                    n_init=20, 
#                    perplexity=30, 
#                    savepath=None, 
#                    title=""):
#     """
#     Perform t-SNE dimensionality reduction and KMeans clustering directly.
#     """
#     n_samples, n_features = emb_matrix.shape
#     print(f"Embedding matrix shape: {emb_matrix.shape} - {n_samples} samples with {n_features} features")
    
#     # Adjust K if too few samples
#     if n_samples < k:
#         k = max(2, n_samples // 2)
#         print(f"Adjusted number of clusters to k={k} due to small sample size")
    
#     # Adjust perplexity
#     if n_samples < 3 * perplexity:
#         perplexity = max(5, n_samples // 3)
#         print(f"Adjusted perplexity to {perplexity} due to small sample size")
    
#     # 1) t-SNE
#     print(f"Running t-SNE with perplexity={perplexity}...")
#     tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
#     coords = tsne.fit_transform(emb_matrix)
    
#     # 2) KMeans on the original features
#     print(f"Running KMeans with k={k}, n_init={n_init}...")
#     kmeans = KMeans(n_clusters=k, random_state=0, n_init=n_init)
#     cluster_labels = kmeans.fit_predict(emb_matrix)
    
#     # Print cluster sizes
#     for cluster_id in range(k):
#         cluster_size = np.sum(cluster_labels == cluster_id)
#         print(f"Cluster {cluster_id+1}: {cluster_size} samples")
    
#     # 3) Plot t-SNE
#     plt.figure(figsize=(10, 8))
#     for cluster_id in range(k):
#         mask = (cluster_labels == cluster_id)
#         plt.scatter(coords[mask, 0], coords[mask, 1],
#                     s=30, alpha=0.7, label=f"Cluster {cluster_id+1}")
    
#     plt.title(f"{title} - t-SNE Visualization")
#     plt.legend()
    
#     if savepath:
#         plt.savefig(f"{savepath}_tsne.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"t-SNE visualization saved to {savepath}_tsne.png")
#     else:
#         plt.show()
    
#     return cluster_labels, coords

# # -------------------------------------------------------------------
# # 2) Spatial clustering visualization
# # -------------------------------------------------------------------
# def spatial_cluster_visualization(df_clusters, positions_df, coords=None, savepath=None, title="Spatial Clustering"):
#     """
#     Create a spatial clustering visualization. If 'spot_id' in df_clusters
#     doesn't match 'spot_id' in positions_df, we fall back to a synthetic layout.
#     """
#     print(f"df_clusters shape: {df_clusters.shape}, spot_id example: {df_clusters['spot_id'].iloc[0]}")
#     print(f"positions_df shape: {positions_df.shape}, spot_id example: {positions_df['spot_id'].iloc[0]}")
    
#     # Check if the spot IDs match your known format
#     if 'batch' in str(df_clusters['spot_id'].iloc[0]) and 'batch' not in str(positions_df['spot_id'].iloc[0]):
#         print("Spot ID formats don't match. Creating synthetic positions for visualization...")
        
#         # Build synthetic positions
#         spot_info_list = []
#         for spot_id in df_clusters['spot_id']:
#             # For example, spot_id = "GTCAGTATGTCCGGCG-1_batch0_sample0_spot0"
#             parts = spot_id.split('_')
            
#             batch_info = [p for p in parts if 'batch' in p]
#             batch_num = int(batch_info[0].replace('batch', '')) if batch_info else 0
            
#             sample_info = [p for p in parts if 'sample' in p]
#             sample_num = int(sample_info[0].replace('sample', '')) if sample_info else 0
            
#             spot_info = [p for p in parts if 'spot' in p]
#             spot_num = int(spot_info[0].replace('spot', '')) if spot_info else 0
            
#             row = batch_num * 12 + sample_num // 10
#             col = (sample_num % 10) * 10 + spot_num
            
#             spot_info_list.append({
#                 'spot_id': spot_id,
#                 'row': row,
#                 'col': col
#             })
        
#         synth_positions = pd.DataFrame(spot_info_list)
#         merged_df = pd.merge(df_clusters, synth_positions, on='spot_id', how='inner')
#     else:
#         # Regular merging
#         merged_df = pd.merge(df_clusters, positions_df, on="spot_id", how="inner")
    
#     print(f"After merging: {len(merged_df)} matching spots")
    
#     if len(merged_df) == 0:
#         print("WARNING: No matching spots found between clusters and positions!")
#         return
    
#     # Figure
#     clusters = sorted(merged_df['cluster'].unique())
#     num_clusters = len(clusters)
#     colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
    
#     plt.figure(figsize=(20, 10))
    
#     # Subplot 1: Spatial distribution
#     plt.subplot(1, 2, 1)
#     for i, cluster in enumerate(clusters):
#         cluster_spots = merged_df[merged_df['cluster'] == cluster]
        
#         # If using real positions, the columns are `row` & `col` or `pxl_row_in_fullres` & `pxl_col_in_fullres`.
#         # Adjust as needed. We'll assume 'row'/'col' are present after merging.
#         plt.scatter(
#             cluster_spots['col'], 
#             cluster_spots['row'],
#             c=[colors[i]],
#             s=50,
#             alpha=0.7,
#             label=f'Cluster {cluster+1}'
#         )
#     plt.legend(loc='upper right')
#     plt.title(f"{title} - Spatial Distribution")
#     plt.xlabel('Column Position')
#     plt.ylabel('Row Position')
#     plt.gca().invert_yaxis()  # optional for top-down
    
#     # Subplot 2: t-SNE if coords are provided
#     if coords is not None:
#         plt.subplot(1, 2, 2)
#         for i, cluster in enumerate(clusters):
#             cluster_idx = np.where(df_clusters['cluster'] == cluster)[0]
#             plt.scatter(
#                 coords[cluster_idx, 0], 
#                 coords[cluster_idx, 1],
#                 c=[colors[i]],
#                 s=50,
#                 alpha=0.7,
#                 label=f'Cluster {cluster+1}'
#             )
        
#         plt.legend(loc='upper right')
#         plt.title(f"{title} - t-SNE Visualization")
#         plt.xlabel('t-SNE Dimension 1')
#         plt.ylabel('t-SNE Dimension 2')
    
#     # Save or show
#     if savepath:
#         plt.savefig(f"{savepath}_spatial.png", dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"Spatial visualization saved to {savepath}_spatial.png")
#     else:
#         plt.show()
# #############################
# # 3) 提取嵌入向量
# #############################
# def extract_inference_embeddings(model, dataloader, device="cuda"):
#     model.eval()
#     all_spot = []
#     all_image = []
#     all_gene = []
    
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(dataloader):
#             print(f"\nProcessing batch {batch_idx+1}/{len(dataloader)}")
            
#             # 安全获取数据
#             imgs = batch[0].to(device)
#             genes = batch[1].to(device)
#             spot_ids = batch[2]  # 列表，包含元组或其他格式
#             row_cols = batch[3].to(device) if len(batch) > 3 else None
            
#             # 打印关键信息用于调试
#             print(f"imgs shape: {imgs.shape}")
#             print(f"genes shape: {genes.shape}")
#             print(f"spot_ids type: {type(spot_ids)}, length: {len(spot_ids)}")
#             print(f"First element of spot_ids: {spot_ids[0]}")
            
#             # 前向传播获取特征
#             img_embs, gene_embs = model(imgs, genes, row_cols)
#             print(f"img_embs shape: {img_embs.shape}")
#             print(f"gene_embs shape: {gene_embs.shape}")
            
#             # 处理每个批次中的每个样本
#             batch_size, num_spots = imgs.shape[:2]
            
#             # 为每个样本和每个spot生成唯一ID
#             for b in range(batch_size):
#                 for s in range(num_spots):
#                     # 创建唯一标识符
#                     if isinstance(spot_ids[0], tuple) and len(spot_ids) >= 1:
#                         # 如果是元组列表，尝试使用第一个元组中的第一个元素
#                         try:
#                             base_id = spot_ids[0][0] if len(spot_ids[0]) > 0 else "unknown"
#                             unique_id = f"{base_id}_batch{batch_idx}_sample{b}_spot{s}"
#                         except:
#                             unique_id = f"batch{batch_idx}_sample{b}_spot{s}"
#                     else:
#                         # 否则使用简单的索引组合
#                         unique_id = f"batch{batch_idx}_sample{b}_spot{s}"
                    
#                     all_spot.append(unique_id)
#                     all_image.append(img_embs[b, s].cpu().numpy())
#                     all_gene.append(gene_embs[b, s].cpu().numpy())
    
#     print(f"Collected features for {len(all_spot)} spots")
    
#     # 转换为矩阵
#     image_emb_matrix = np.array(all_image)
#     gene_emb_matrix = np.array(all_gene)
    
#     # 手动计算融合特征 (简单平均)
#     fuse_emb_matrix = (image_emb_matrix + gene_emb_matrix) / 2
    
#     # 创建DataFrame
#     df_image = pd.DataFrame(image_emb_matrix)
#     df_image["spot_id"] = all_spot
    
#     df_gene = pd.DataFrame(gene_emb_matrix)
#     df_gene["spot_id"] = all_spot
    
#     df_fuse = pd.DataFrame(fuse_emb_matrix)
#     df_fuse["spot_id"] = all_spot
    
#     return df_image, df_gene, df_fuse

# #############################
# # 4) 主函数
# #############################
# def main():
#     #############################
#     # A. 路径和参数设置
#     #############################
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     out_dir = "/home/nsong/Craig_SPA6_A"
#     root_dir = "/home/nsong/Craig_SPA6_A"
#     metadata_csv = os.path.join(root_dir, "metadata.csv")
#     tissue_position_csv = os.path.join(out_dir, "spatial", "tissue_positions.csv")
#     model_ckpt = "checkpoints/longnet_20250305_010027/best_model.pth"  # 使用您的最佳模型
    
#     # 聚类和可视化参数
#     batch_size = 16
#     k_clusters = 5
#     perplexity = 30
#     num_spots_per_sample = 10  # 每个样本的spot数量

#     #############################
#     # B. 数据加载
#     #############################
#     transform_img = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     # 创建用于推理的数据集
#     inference_dataset = STMultiModalDataset(
#         metadata_csv=metadata_csv,
#         tissue_positions_csv=tissue_position_csv,
#         root_dir=root_dir,
#         transform_image=transform_img,
#         num_spots_per_sample=num_spots_per_sample,
#         mode='val'
#     )
    
#     loader = DataLoader(
#         inference_dataset, 
#         batch_size=batch_size, 
#         shuffle=False,
#         num_workers=4
#     )
    
#     # 加载空间位置信息
#     if tissue_position_csv.endswith('.parquet'):
#         import pyarrow.parquet as pq
#         positions_df = pq.read_table(tissue_position_csv).to_pandas()
#         positions_df.rename(columns={"barcode": "spot_id"}, inplace=True)
#     else:
#         positions_df = pd.read_csv(tissue_position_csv, header=None)
#         positions_df.columns = ["spot_id","in_tissue","row","col","pxl_row_in_fullres","pxl_col_in_fullres"]
    
#     print(f"Loaded position information for {len(positions_df)} spots")

#     #############################
#     # C. 加载模型
#     #############################
#     # 创建图像编码器
#     img_encoder = ImageEncoder(
#         model_name="vit_small_patch16_224_dino", 
#         pretrained=True,
#         embed_dim=256,
#         freeze=False
#     )
    
#     # 创建基因编码器
#     gene_encoder = SpotLongNetGeneEncoder(
#         gene_dim=2000,
#         embed_dim=256,
#         config_name="LongNet_for_spatial"
#     )
    
#     # 创建多模态模型
#     model = MultiModalModel(img_encoder, gene_encoder, proj_dim=128)
    
#     # 加载模型权重
#     state_dict = torch.load(model_ckpt, map_location=device)
#     model.load_state_dict(state_dict)
#     model = model.to(device)
#     print(f"Model loaded from {model_ckpt}")

#     #############################
#     # D. 提取嵌入向量
#     #############################
#     df_image, df_gene, df_fuse = extract_inference_embeddings(model, loader, device=device)
#     print(f"Extracted embeddings for {len(df_fuse)} spots")
    
#     # 去除spot_id列获取特征矩阵
#     image_matrix = df_image.drop("spot_id", axis=1).values
#     gene_matrix = df_gene.drop("spot_id", axis=1).values
#     fuse_matrix = df_fuse.drop("spot_id", axis=1).values

#     #############################
#     # E. 聚类和可视化
#     #############################
#     # 1) 图像特征
#     print("\nProcessing image embeddings...")
#     image_labels, image_coords = do_tsne_kmeans(
#         emb_matrix=image_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath="longnet_image",
#         title="LongNet Image Embeddings"
#     )
#     df_image_cluster = pd.DataFrame({
#         "spot_id": df_image["spot_id"],
#         "cluster": image_labels
#     })
#     df_image_cluster.to_csv(os.path.join(out_dir, "longnet_image_clusters.csv"), index=False)
    
#     # 空间可视化
#     spatial_cluster_visualization(
#         df_image_cluster, 
#         positions_df, 
#         coords=image_coords,
#         savepath="longnet_image",
#         title="LongNet Image Embeddings"
#     )

#     # 2) 基因特征
#     print("\nProcessing gene embeddings...")
#     gene_labels, gene_coords = do_tsne_kmeans(
#         emb_matrix=gene_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath="longnet_gene",
#         title="LongNet Gene Embeddings"
#     )
#     df_gene_cluster = pd.DataFrame({
#         "spot_id": df_gene["spot_id"],
#         "cluster": gene_labels
#     })
#     df_gene_cluster.to_csv(os.path.join(out_dir, "longnet_gene_clusters.csv"), index=False)
    
#     # 空间可视化
#     spatial_cluster_visualization(
#         df_gene_cluster, 
#         positions_df, 
#         coords=gene_coords,
#         savepath="longnet_gene",
#         title="LongNet Gene Embeddings"
#     )

#     # 3) 融合特征
#     print("\nProcessing fused embeddings...")
#     fused_labels, fused_coords = do_tsne_kmeans(
#         emb_matrix=fuse_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath="longnet_fused",
#         title="LongNet Fused Embeddings"
#     )
#     df_fused_cluster = pd.DataFrame({
#         "spot_id": df_fuse["spot_id"],
#         "cluster": fused_labels
#     })
#     df_fused_cluster.to_csv(os.path.join(out_dir, "longnet_fused_clusters.csv"), index=False)
    
#     # 空间可视化
#     spatial_cluster_visualization(
#         df_fused_cluster, 
#         positions_df, 
#         coords=fused_coords,
#         savepath="longnet_fused",
#         title="LongNet Fused Embeddings"
#     )
    
#     print("All visualizations completed!")

# if __name__=="__main__":
#     main()




# import os
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sys

# from torch.utils.data import DataLoader
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# from dataset import STMultiModalDataset
# from model import MultiModalModel, ImageEncoder, GeneEncoder
# # from LongNet import make_longnet_from_name, twoD_sin_cos_emb

# #############################
# # 1) K-Means + t-SNE 
# #############################

# def do_tsne_and_kmeans(emb_matrix, k=5, perplexity=30, savepath=None, title=""):
#     kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
#     cluster_labels = kmeans.fit_predict(emb_matrix)  # [N,]

#     tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
#     coords = tsne.fit_transform(emb_matrix)  # [N, 2]

#     plt.figure(figsize=(6,6))
#     for cluster_id in range(k):
#         mask = (cluster_labels == cluster_id)
#         plt.scatter(coords[mask,0], coords[mask,1], s=5, alpha=0.7, label=f"Cluster {cluster_id+1}")

#     plt.title(title)
#     plt.legend()
#     if savepath:
#         plt.savefig(savepath, dpi=300)
#         plt.close()
#         print(f"[{title}] t-SNE with K-Means(k={k}) saved to {savepath}")
#     else:
#         plt.show()

#     return cluster_labels
# def do_pca_tsne_kmeans(emb_matrix, 
#                        k=5, 
#                        n_components_pca=20, 
#                        n_init=50, 
#                        perplexity=30, 
#                        savepath=None, 
#                        title=""):
    

#     # 1) PCA 降维 => [N, n_components_pca]
#     pca = PCA(n_components=n_components_pca, random_state=42)
#     pca_emb = pca.fit_transform(emb_matrix)  

#     # 2) KMeans => 在 PCA 后的 20 维上聚类
#     kmeans = KMeans(n_clusters=k, random_state=0, n_init=n_init)
#     cluster_labels = kmeans.fit_predict(pca_emb)  # [N,]

#     # 3) t-SNE => 在 PCA 后的特征上做2D可视化
#     tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
#     coords = tsne.fit_transform(pca_emb)  # [N, 2]

#     # 4) 画图
#     plt.figure(figsize=(6,6))
#     for cluster_id in range(k):
#         mask = (cluster_labels == cluster_id)
#         plt.scatter(coords[mask, 0], coords[mask, 1],
#                     s=5, alpha=0.7, label=f"Cluster {cluster_id+1}")

#     plt.title(title)
#     plt.legend()
#     if savepath:
#         plt.savefig(savepath, dpi=300)
#         plt.close()
#         print(f"[{title}] PCA->t-SNE with K-Means(k={k}, n_init={n_init}), perplexity={perplexity} saved to {savepath}")
#     else:
#         plt.show()

#     return cluster_labels

# #############################
# # 2) embedding
# #############################
# def extract_inference_embeddings(model, dataloader, device="cuda"):
#     model.eval()
#     all_gene = []
#     all_fuse = []
#     all_spot = []
#     all_image = []

#     with torch.no_grad():
#         # for (img, gene, spot_ids, row_col) in dataloader:
#         for (img, gene, spot_ids) in dataloader:
#             img = img.to(device)
#             gene= gene.to(device)
#             # row_col = row_col.to(device)

#             # forward => (img_emb, gene_emb, fused_emb)
#             # img_emb, gene_emb, fused_emb = model(img, gene, rowcol=row_col)
#             img_emb, gene_emb, fused_emb = model(img, gene)

#             all_image.append(img_emb.cpu())
#             all_gene.append(gene_emb.cpu())
#             all_fuse.append(fused_emb.cpu())
#             all_spot.extend(spot_ids)

#     image_emb_matrix = torch.cat(all_image, dim=0).numpy()
#     gene_emb_matrix  = torch.cat(all_gene, dim=0).numpy()
#     fuse_emb_matrix  = torch.cat(all_fuse, dim=0).numpy()

#     df_image = pd.DataFrame(image_emb_matrix)
#     df_image["spot_id"] = all_spot
    
#     df_gene = pd.DataFrame(gene_emb_matrix)
#     df_gene["spot_id"] = all_spot

#     df_fuse = pd.DataFrame(fuse_emb_matrix)
#     df_fuse["spot_id"] = all_spot

#     return df_image, df_gene, df_fuse

# def main():
#     #############################
#     # A. Path
#     #############################
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     root_dir = "/coh_labs/dits/nsong/ovarian/outs"
#     tissue_position_csv = os.path.join(root_dir, "spatial/tissue_positions.csv")
#     metadata_csv = os.path.join(root_dir, "metadata.csv")
#     # tissue_position_csv = os.path.join("/coh_labs/dits/nsong/Mayo_VisiumHD/101/outs", "binned_outputs", "square_016um", "spatial", "tissue_positions.parquet")
#     model_ckpt   = "sthistoclip_model_j13_ori.pth" 
#     batch_size   = 128
#     k_clusters   = 5
#     perplexity   = 30
#     out_dir = "/coh_labs/dits/nsong/ovarian/outs/results_ori"
#     os.makedirs(out_dir, exist_ok=True)

#     #############################
#     # B. Dataloader
#     #############################
#     from dataset import STMultiModalDataset
#     import torchvision.transforms as T

#     transform_img = T.Compose([
#         # T.ToPILImage(),
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#     ])

#     dataset = STMultiModalDataset(
#         metadata_csv=metadata_csv,
#         root_dir=root_dir,
#         tissue_positions_csv=tissue_position_csv,
#         transform_image=transform_img
#     )
    
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     #############################
#     # C. load model
#     #############################
#     img_encoder = ImageEncoder(
#         model_name="vit_small_patch16_224_dino", 
#         pretrained=True,
#         out_dim=256, freeze=False
#     )
#     # gene_encoder = SpotLongNetGeneEncoder(
#     #     config_name="LongNet_8_layers_256_dim",
#     #     gene_dim=1000,       
#     #     out_dim=256,
#     #     do_sin_cos=True      
#     # )
#     gene_encoder =GeneEncoder(input_dim=2000, hidden_dim=512, out_dim=256)
#     model = MultiModalModel(img_encoder, gene_encoder, proj_dim=128)
#     state_dict = torch.load(model_ckpt, map_location=device)
#     model.load_state_dict(state_dict)
#     model = model.to(device)

#     #############################
#     # D. get "post_image", "post_gene", "fused_emb"
#     #############################
#     df_image, df_gene, df_fuse = extract_inference_embeddings(model, loader, device=device)
#     print(df_fuse.shape)
#     image_matrix = df_image.drop("spot_id", axis=1).values
#     gene_matrix  = df_gene.drop("spot_id", axis=1).values
#     fuse_matrix  = df_fuse.drop("spot_id", axis=1).values

#     # 1) post_image + KMeans + t-SNE
#     post_image_labels = do_tsne_and_kmeans(
#         emb_matrix=image_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath="post_image_kmeans_tsne.png",
#         title="Trained Image Embedding with KMeans"
#     )
#     df_postimage_cluster = pd.DataFrame({
#         "spot_id": df_image["spot_id"],
#         "cluster": post_image_labels
#     })
#     df_postimage_cluster.to_csv(os.path.join(out_dir,"post_image_clusters.csv"), index=False)
#     print("Saved post_image_clusters.csv")

#     # 2) post_gene + KMeans + t-SNE
#     post_gene_labels = do_tsne_and_kmeans(
#         emb_matrix=gene_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath="post_gene_kmeans_tsne.png",
#         title="Trained Gene Embedding with KMeans"
#     )
#     df_postgene_cluster = pd.DataFrame({
#         "spot_id": df_gene["spot_id"],
#         "cluster": post_gene_labels
#     })
#     df_postgene_cluster.to_csv(os.path.join(out_dir,"post_gene_clusters.csv"), index=False)
#     print("Saved post_gene_clusters.csv")

#     # 3) fused_emb + KMeans + t-SNE
#     fused_labels = do_tsne_and_kmeans(
#         emb_matrix=fuse_matrix,
#         k=k_clusters,
#         perplexity=perplexity,
#         savepath="fused_kmeans_tsne.png",
#         title="Fused Embedding (Gene+Image) with KMeans"
#     )
#     df_fused_cluster = pd.DataFrame({
#         "spot_id": df_fuse["spot_id"],
#         "cluster": fused_labels
#     })
#     df_fused_cluster.to_csv(os.path.join(out_dir,"fused_clusters.csv"), index=False)
#     print("Saved fused_clusters.csv")

# if __name__=="__main__":
#     main()

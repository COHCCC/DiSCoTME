import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import timm

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import timm

# 使用现有metadata的数据集类
class MetadataImageDataset(Dataset):
    def __init__(self, metadata_path, root_dir, transform=None):
        """
        从现有的metadata文件加载数据
        
        Args:
            metadata_path: metadata.csv文件路径
            root_dir: 根目录，用于组合完整的图像路径
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 读取metadata
        self.metadata = pd.read_csv(metadata_path)
        print(f"Loaded metadata with {len(self.metadata)} entries")
        
        # 检查metadata中是否有必要的列
        required_columns = ["spot_id", "image_path"]
        for col in required_columns:
            if col not in self.metadata.columns:
                raise ValueError(f"Required column '{col}' not found in metadata")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        spot_id = row["spot_id"]
        
        # 获取图像路径
        img_path = os.path.join(self.root_dir, row["image_path"])
        
        # 读取图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 如果图像加载失败，返回一个黑色图像
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "target_id": spot_id
        }

############################
# PCA + t-SNE + KMeans
############################
def do_pca_tsne_kmeans(emb_matrix, 
                       k=5, 
                       n_components_pca=20,
                       n_init=50, 
                       perplexity=30, 
                       savepath=None, 
                       title=""):

    pca = PCA(n_components=n_components_pca, random_state=42)
    pca_emb = pca.fit_transform(emb_matrix)
    
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=n_init)
    cluster_labels = kmeans.fit_predict(pca_emb)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_coords = tsne.fit_transform(pca_emb)
    
    plt.figure(figsize=(6, 6))
    for cid in range(k):
        mask = (cluster_labels == cid)
        plt.scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                    s=5, alpha=0.7, label=f"Cluster {cid+1}")
    plt.title(title)
    plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=300)
        plt.close()
        print(f"[{title}] PCA->t-SNE with KMeans (k={k}) saved to {savepath}")
    else:
        plt.show()
    
    return cluster_labels, tsne_coords

def extract_dino_embeddings(model, dataloader, device="cuda"):
    """
    使用DINO v2模型提取图像特征
    """
    model.eval()
    all_img = []
    all_spot = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            spot_ids = batch["target_id"]
            
            # 使用DINO模型提取特征
            img_emb = model(images)
            
            all_img.append(img_emb.cpu())
            all_spot.extend(spot_ids)

    img_matrix = torch.cat(all_img, dim=0).numpy()
    
    df_img = pd.DataFrame(img_matrix)
    df_img["spot_id"] = all_spot
    
    return df_img

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 设置路径和参数 - 根据实际情况修改
    root_dir = "/coh_labs/dits/Craig_Spatial/Craig_SPA8_A/outs"  # 根目录
    metadata_path = os.path.join(root_dir, "metadata.csv")  # metadata文件路径
    batch_size = 16
    perplexity = 30
    
    # 设置输出目录
    out_dir = os.path.join(root_dir, "results/dino_v2_direct")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # 设置图像变换
    transform_image = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    dataset = MetadataImageDataset(
        metadata_path=metadata_path,
        root_dir=root_dir,
        transform=transform_image
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 加载预训练的DINO v2模型
    model = timm.create_model("vit_small_patch16_224_dino", pretrained=True)
    
    # 修改模型，只输出特征
    class DinoFeatureExtractor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # 获取DINO的特征表示
            features = self.model.forward_features(x)
            # 使用CLS token作为图像表示
            return features[:, 0]
    
    feature_extractor = DinoFeatureExtractor(model)
    feature_extractor = feature_extractor.to(device)
    
    # 提取特征
    print("Extracting DINO v2 embeddings...")
    df_img = extract_dino_embeddings(feature_extractor, dataloader, device=device)
    print(f"Extracted embeddings shape: {df_img.shape}")
    
    # 保存嵌入
    img_csv = os.path.join(out_dir, "dino_v2_embeddings.csv")
    df_img.to_csv(img_csv, index=False)
    print(f"Saved DINO v2 embeddings CSV to {img_csv}")
    
    # 获取嵌入矩阵
    image_matrix = df_img.drop("spot_id", axis=1).values
    
    # 对不同的k值执行聚类和保存结果
    all_cluster_results = {}
    all_tsne_coords = {}
    
    for k in range(4, 11):
        print(f"\n--- Processing clustering with k={k} ---")
        
        k_dir = os.path.join(out_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)
        print(f"Created directory: {k_dir}")
        
        # 执行PCA、t-SNE和KMeans聚类，不在文件名中包含k值
        image_labels, tsne_coords = do_pca_tsne_kmeans(
            emb_matrix=image_matrix,
            k=k,
            perplexity=perplexity,
            savepath=os.path.join(k_dir, "dino_v2_kmeans_tsne.png"),
            title=f"DINO v2 Image Embedding with KMeans (k={k})"
        )
        
        # 保存聚类结果
        df_image_cluster = pd.DataFrame({
            "spot_id": df_img["spot_id"],
            "cluster": image_labels
        })
        
        # 保存聚类结果CSV，不在文件名中包含k值
        image_cluster_csv = os.path.join(k_dir, "dino_v2_clusters.csv")
        df_image_cluster.to_csv(image_cluster_csv, index=False)
        print(f"Saved DINO v2 clusters CSV to {image_cluster_csv}")
        
        # 将结果添加到字典中，以便后续可能的组合可视化
        all_cluster_results[k] = image_labels
        all_tsne_coords[k] = tsne_coords
    
    # 保存所有聚类结果的组合CSV
    combined_results = pd.DataFrame({"spot_id": df_img["spot_id"]})
    for k in range(4, 11):
        combined_results[f"cluster_k{k}"] = all_cluster_results[k]
    
    combined_csv = os.path.join(out_dir, "dino_v2_all_clusters.csv")
    combined_results.to_csv(combined_csv, index=False)
    print(f"Saved combined cluster results to {combined_csv}")
    
    print("\nAll processing completed!")
    
if __name__ == "__main__":
    main()
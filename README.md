# DiSCoTME (Dilated Spatial Context for the Tumor Microenvironment)

## Using multi-scale spatial integration of histology and transcriptomics to discover hidden tumor microenvironment motifs

Jiarong Song1, Bohan Zhang1, Rayyan Aburajab1, Jing Qian2, Rania Bassiouni1, John J.Y. Lee1, John D. Carpten1, David W. Craig1*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Model Overview
**DiSCoTME** (Dilated Spatial Context for the Tumor Microenvironment) is a dual-encoder deep learning framework that jointly learns from H&E histology images and gene expression profiles in spatial transcriptomics data. By combining multi-scale dilated neighbor attention with contrastive learning, DiSCoTME produces unified, spatially informed embeddings that capture tissue organization across multiple spatial scales.

## Key Features

- **Multi-modal integration**: Contrastive learning aligns spatially-aware image and gene embeddings into a shared space
- **Multi-scale spatial context**: Multi-head neighbor attention with varying dilation rates resolve from local coposition to micro-anatomical structures to global context
- **Adaptive gating**: Per-spot coefficients balance local identity against neighborhood context where high values preserve sharp local gradients in distinctive regions (fidelity mode), low values draw on neighborhood consistency when local evidence is sparse (denoising mode)
- **Unsupervised discovery**: Joint embeddings encode cross-modal correspondence and neighborhood context, enabling unsupervised clustering to reveal hidden microenvironmental motifs defined by the interplay of all three dimensions without manual annotation

## Input / Output

- **Input**: 10x Visium spatial transcriptomics (gene expression + H&E image + spot coordinates)
- **Output**: Spatially-aware joint embeddings for unsupervised motif discovery—revealing tissue structures

<p align="center">
  <img src="assets/overview.png" width="90%"> <br>

  *Overview of DiSCoTME model architecture*

</p>

## Download Test Data
```bash
# Transcriptome data from GEO
wget -r -np -nd ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8513nnn/GSM8513873/suppl/

# H&E image from Hugging Face
wget https://huggingface.co/datasets/nina-song/SPA1_D/resolve/main/Craig_SPA1_D.tif
```

## Installation
1. Create Environment
```bash
conda create -n discotme python=3.9.18
conda activate discotme
git clone https://github.com/COHCCC/DiSCoTME.git
cd DiSCoTME
```

2. Install core packages and dependencies:
```bash
# Install PyTorch with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install timm
pip install timm==1.0.22

# Install torchscale from source (required for dilated attention)
# Option A: If pip install git+... works on your system:
pip install git+https://github.com/microsoft/torchscale.git@4d1e0e82e5adf86dd424f1463192635b73fc8efc --no-deps

# Option B: If git is not available to pip, install manually:
git clone https://github.com/microsoft/torchscale.git /tmp/torchscale
cd /tmp/torchscale
git checkout 4d1e0e82e5adf86dd424f1463192635b73fc8efc
pip install . --no-deps
cd -

# Install remaining dependencies
pip install -r requirements.txt
```

## Data Preprocessing

This script prepares Visium datasets for downstream training by aligning high-resolution WSI images with spatial transcriptomics data. It crops image patches for each spot and generates corresponding gene expression vectors.

### Usage
```bash
# Option A: Default HVG selection (using log-normalized data)
python scripts/run_preprocess.py --root /path/to/spaceranger_outs/ --n_genes 2000

# Option B: Seurat_v3 HVG selection (using raw counts)
python scripts/run_preprocess.py --root /path/to/spaceranger_outs/ --hvg_method seurat_v3 --n_genes 3000
```

### Arguments

| Argument | Required | Description |
| :--- | :--- | :--- |
| `--root` | **Yes** | Path to Visium directory (must contain `.h5` file and `spatial/` folder). |
| `--hvg_method` | No | Selection flavor: default or seurat_v3. |
| `--n_genes` | No | Number of highly variable genes to retain. (Default: 2000). |
| `--wsi` | No | Path to WSI image (.tif). Auto-detected if not specified. |
| `--radius` | No | Patch radius in pixels. Auto-inferred from `scalefactors_json.json` if omitted. |
| `--out` | No | Output directory. Defaults to `--root`. |

### Output Structure
```
output_dir/
├── metadata.csv        # spot_id, image_path, gene_vector_path
├── image/
│   ├── BARCODE1.jpg
│   ├── BARCODE2.jpg
│   └── ...
└── genes/
    ├── BARCODE1.npy
    ├── BARCODE2.npy
```

## Internal testing: SLURM Cluster (will be removed before submission)

For City of Hope HPC users and internal test, see `scripts/run_ddp.sh` as a template.

Key modifications needed:
- `--partition`: Your cluster's GPU partition
- `--gres`: GPU type (e.g., `gpu:v100-dev:4`)
- Conda environment path
- Data paths
```bash
# Copy and modify for your cluster
cp scripts/run_ddp.sh scripts/run_my_cluster.sh
```
**important.** replace dataset name
```
DATASET_NAME="SPA_1D"
DATA_ROOT="/path/to/dataset/${DATASET_NAME}"
```
# Edit the script, then submit
```
sbatch scripts/run_my_cluster.sh
```


## Quick Start
```bash
# 1. Copy config and change data path
cp configs/default.yaml configs/my_config.yaml
# Edit my_config.yaml: set data.root to your data folder

## Training
### Single GPU
```bash
python scripts/run_train.py --config configs/my_config.yaml
```

### Multi-GPU (Single Node)
```bash
torchrun --nproc_per_node=4 scripts/run_train.py --config configs/my_config.yaml
```

### Command Line Override

Any config value can be overridden via command line:
```bash
torchrun --nproc_per_node=4 scripts/run_train.py \
    --config configs/default.yaml \
    --batch-size 16 \
    --num-epochs 100 \
    --temperature 0.05
```

---

## Key Parameters

### Most Commonly Tuned

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Temperature | `--temperature` | 0.07 | InfoNCE temperature. Lower = sharper distribution |
| Batch size | `--batch-size` | 12 | Per-GPU batch size |
| Epochs | `--num-epochs` | 50 | Training epochs |
| Local neighbors | `--num-local` | 15 | Spatial neighbors per spot |

### Learning Rates

Different components use different learning rates for optimal training:

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Image backbone | `--lr-img-backbone` | 1e-5 | Pretrained ViT/GigaPath, tune slowly |
| Image projection | `--lr-img-proj` | 1e-4 | Projection layers |
| Image context | `--lr-img-context` | 1e-4 | Dilated attention |
| Gene encoder | `--lr-gene-encoder` | 3e-4 | Learns from scratch, can be faster |
| Gene projection | `--lr-gene-proj` | 1e-4 | Final projection |

Example - adjust learning rates:
```bash
torchrun --nproc_per_node=4 scripts/run_train.py \
    --config configs/default.yaml \
    --lr-img-backbone 5e-6 \
    --lr-gene-encoder 1e-4 \
    --temperature 0.1
```

---

## Available Modules

### Model Architectures (`--model-arch`)

| Name | Description |
|------|-------------|
| `standard_discotme` | **Default.** Contrastive learning with gated fusion |


### Image Encoders (`--image-encoder-type`)

| Name | Description |
|------|-------------|
| `gated_image_encoder` | **Default.** Adaptive gate for identity-context fusion |


### Image Backbones (`--image-backbone`)

| Name | Model | Speed | Notes |
|------|-------|-------|-------|
| `vit_dino_v1` | ViT-S/16 DINO | ⚡ Fast | **Default.** Good for most cases |
| `gigapath_frozen_v1` | GigaPath | 🐢 Slow | Pathology-pretrained, requires HF token |


### Gene Encoders (`--gene-encoder-type`)

| Name | Description |
|------|-------------|
| `gated_gene_encoder` | **Default.** MLP + residual + adaptive gate |
| `gene_mlp_residual` | MLP with full residual connection |
| `gene_mlp_weighted_residual` | MLP with 0.1 weighted residual |
| `gene_transformer_intra` | Transformer-based encoder |
| `gene_identity_transformer` | Per-gene embedding + transformer |
| `gene_simple_linear` | Simple linear projection (baseline) |

### Dilated Attention Presets (`--context-config`)

| Name | Layers | Heads | Use Case |
|------|--------|-------|----------|
| `LongNet_for_spatial` | 4 | 8 | **Default.** Standard Visium |
| `LongNet_for_large_spatial` | 8 | 16 | Visium HD / large datasets |
| `LongNet_8_layers_256_dim` | 8 | 16 | Deep model |

---

## Configuration

### Minimal Config
```yaml
data:
  root: "/path/to/your/data"
```

### Full Config
```yaml
# configs/default.yaml
data:
  root: "/path/to/your/data"
  metadata_csv: "metadata.csv"
  tissue_positions_csv: "tissue_positions.csv"
  num_local: 15
  num_global: 0
  local_distance: 400

model:
  arch: "standard_discotme"
  image_encoder_type: "gated_image_encoder"
  image_backbone: "vit_dino_v1"
  gene_encoder_type: "gated_gene_encoder"
  embed_dim: 256
  proj_dim: 128

context:
  preset: "LongNet_for_spatial"

training:
  batch_size: 12
  num_epochs: 50
  temperature: 0.07
  weight_decay: 1e-5
  seed: 42
  use_distill: false
  distill_weight: 0.0

lr:
  img_backbone: 1e-5
  img_proj: 1e-4
  img_context: 1e-4
  gene_encoder: 3e-4
  gene_proj: 1e-4
  default: 1e-4

output:
  save_dir: "checkpoints"
  run_name: null
```

### Custom Dilated Attention
```yaml
context:
  preset: null  # disable preset
  custom:
    encoder_layers: 6
    encoder_embed_dim: 256
    encoder_ffn_embed_dim: 1024
    encoder_attention_heads: 8
    dilated_ratio: "[1, 2, 4, 8]"
    segment_length: "[1000, 2000, 4000, 8000]"
    dropout: 0.1
    drop_path_rate: 0.1
```


## Internal testing: SLURM Cluster

For City of Hope HPC users and internal test, see `scripts/run_ddp.sh` as a template.

Key modifications needed:
- `--partition`: Your cluster's GPU partition
- `--gres`: GPU type (e.g., `gpu:v100-dev:4`)
- Conda environment path
- Data paths
```bash
# Copy and modify for your cluster
cp scripts/run_ddp.sh scripts/run_my_cluster.sh
```
**important.** replace dataset name
```
DATASET_NAME="SPA_1D"
DATA_ROOT="/path/to/dataset/${DATASET_NAME}"
```
# Edit the script, then submit
```
sbatch scripts/run_my_cluster.sh
```

---

## Outputs

Training outputs are saved to `checkpoints/<run_name>/`:
```
checkpoints/standard_discotme_20250123_143052/
├── config.yaml           # Full config (for reproducibility)
├── best_model.pth        # Best model weights
├── final_model.pth       # Final model weights
├── checkpoint_epoch5.pth # Periodic checkpoints
├── checkpoint_epoch10.pth
└── loss_history.json     # Training loss curve
```

## Model Evaluation & Interpretation
After training, you can use the following script to perform inference, generate spatial clusters, and extract the Alpha weights

1. Run Evaluation
This script will extract multi-modal embeddings from your Visium data and perform K-Means clustering across multiple scales.
```bash
python run_eval_with_alpha.py \
    --data-root /path/to/your/visium_data/ \
    --model-path /path/to/checkpoints/best_model.pth \
    --output-dir ./eval_results/ \
    --device cuda \
    --batch-size 64 \
    --num-workers 4
```

| Name | Description |
|------|-------------|
| `data-root` | Path to the directory containing tissue_positions.csv and image patches |
| `model-path` | **important.** Must point to the specific .pth file, not just the folder. |
| `output-dir` | Where clustering results and alpha weights will be saved |
| `device` | Set to cuda to leverage GPU |

2. Expected Outputs
* k[4-10]/: Directories containing .csv cluster assignments for post-training embeddings (Image, Gene, and Fused (more details about fusing can be found in online method section)).
* alpha_weights_e5.csv: Learned alpha weights per spot. A per-spot gating coefficient preserves sharp local gradients for distinct motifs (fidelity mode) while smoothing noise in homogeneous regions (denoising mode).

## Citation

If you find DiSCoTME useful in your research, please cite:
```
Song, N., & Craig, D. (2026). Using multi-scale spatial integration of histology and transcriptomics to discover hidden tumor microenvironment motifs
```
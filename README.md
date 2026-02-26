# DiSCoTME (Dilated Spatial Context for the Tumor Microenvironment)

## Using multi-scale spatial integration of histology and transcriptomics to discover hidden tumor microenvironment motifs

Jiarong Song, Bohan Zhang, Rayyan Aburajab, Jing Qian, Rania Bassiouni, John J.Y. Lee, John D. Carpten, David W. Craig*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Model Overview
**DiSCoTME** (Dilated Spatial Context for the Tumor Microenvironment) is a dual-encoder deep learning framework that jointly learns from H&E histology images and gene expression profiles in spatial transcriptomics data. By combining multi-scale dilated neighbor attention with contrastive learning, DiSCoTME produces unified, spatially informed embeddings that capture tissue organization across multiple spatial scales.

## Key Features

- **Multi-modal integration**: Contrastive learning aligns spatially-aware image and gene embeddings into a shared space
- **Multi-scale spatial context**: Multi-head neighbor attention with varying dilation rates resolve from local coposition to micro-anatomical structures to global context
- **Adaptive gating**: Per-spot coefficients balance local identity against neighborhood context where high values preserve sharp local gradients in distinctive regions (fidelity mode), low values draw on neighborhood consistency when local evidence is sparse (denoising mode)
- **Unsupervised discovery**: Joint embeddings encode cross-modal correspondence and neighborhood context, enabling unsupervised clustering to reveal hidden microenvironmental motifs defined by the interplay of all three dimensions without manual annotation

## Input / Output

| | Description |
|---|---|
| **Input** | 10x Visium spatial transcriptomics data with matched H&E whole slide image and spot coordinates |
| **Output** | Joint histology-gene embeddings that capture cross-modal correspondence and multi-scale spatial context, enabling unsupervised discovery of microenvironmental motifs invisible to either modality alone

<p align="center">
  <img src="assets/github.png" width="95%"> <br>

  *Overview of DiSCoTME model architecture*

</p>

# Download Test Data

**Demo Data**: This is a human glioblastoma (GBM) spatial transcriptomics dataset from GEO ([GSM8513873](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8513873), part of series [GSE242352](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242352)), generated using the 10X Genomics Visium CytAssist platform on FFPE tissue sections. The data includes gene expression matrices (filtered/raw H5 files), spatial coordinates, tissue images, and visualization files, enabling exploration of spatial heterogeneity in the tumor microenvironment.

```bash
# Transcriptome data from GEO
mkdir -p demo_data
wget -r -np -nd -P demo_data ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8513nnn/GSM8513873/suppl/

cd demo_data
gunzip *.gz

# Remove unnecessary headings
for f in GSM8513873_SPA1_D_*; do
    mv "$f" "${f#GSM8513873_SPA1_D_}"
done

mkdir -p spatial
mv tissue_positions.csv spatial/
mv scalefactors_json.json spatial/

# H&E image from Hugging Face
wget https://huggingface.co/datasets/nina-song/SPA1_D/resolve/main/Craig_SPA1_D.tif
```
## Required Input Data Structure

**Note:** If using your own data, ensure it follows the same structure as the demo data. Specifically:
- All files should have no extra prefix characters
- `tissue_positions.csv` and `scalefactors_json.json` must be placed inside the `spatial/` folder
- File names should match exactly as shown above (except .tif for whole slide H&E image)
- *If you encounter file format errors during preprocessing, check all files in this folder to ensure consistent formatting for downstream processing and training*

```
demo_data/
├── filtered_feature_bc_matrix.h5
├── WSI.tif
├── spatial/
│   ├── tissue_positions.csv
│   └── scalefactors_json.json
└── ...
```

# Installation
1. Create Environment
```bash
conda create -n discotme python=3.9.18
conda activate discotme
git clone https://github.com/COHCCC/DiSCoTME.git
cd DiSCoTME
```

2. Install core packages:
```bash
# Install PyTorch with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# Install timm
pip install timm==1.0.22
```
3. Install torchscale

**Option A: Standard installation** (works on most systems)
```bash
pip install git+https://github.com/microsoft/torchscale.git@4d1e0e82e5adf86dd424f1463192635b73fc8efc --no-deps
```

**Option B: Manual installation** (for HPC environments where git runs inside Singularity/container)
```bash
git clone https://github.com/microsoft/torchscale.git /tmp/torchscale
cd /tmp/torchscale
git checkout 4d1e0e82e5adf86dd424f1463192635b73fc8efc
pip install . --no-deps

# Back to the previous DiSCoTME folder
cd - 
```

4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

> **Note**: You may see a warning/error message about `torchscale 0.2.0 requires timm==0.6.13`. This can be safely ignored as we install torchscale with `--no-deps` to use our required timm version.

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
├── metadata.csv        # column name: spot_id, image_path, gene_vector_path
├── image/
│   ├── BARCODE1.jpg
│   ├── BARCODE2.jpg
│   └── ...
└── genes/
    ├── BARCODE1.npy
    ├── BARCODE2.npy
```

# Internal testing: SLURM Cluster (will be REMOVED BEFORE submission)

For HPC users running on SLURM-managed clusters:

1. **Copy and edit the template:**

```bash
cp scripts/run_train_slurm.sh scripts/run_my_cluster.sh
```

2. **Modify the USER CONFIGURATION section at the top:**

```bash
# =============================================================================
# USER CONFIGURATION - MODIFY THESE
# =============================================================================

PROJECT_ROOT="/path/to/DiSCoTME"      # Full path to your DiSCoTME directory
CONDA_ENV="discotme"                   # Your conda environment name
DATA_ROOT="/path/to/your/data"         # Full path to preprocessed data
META_CSV="metadata.csv"                # Metadata filename
POS_CSV="tissue_positions.csv" # Tissue positions path
```

3. **Adjust SLURM settings as needed:**

```bash
#SBATCH --partition=gpu-v100-dev         # Your cluster's GPU partition
#SBATCH --gres=gpu:4             # Number of GPUs (e.g., gpu:v100:4)
#SBATCH --mail-user=your@email   # Your email for notifications
```

4. **Submit the job:**

```bash
sbatch scripts/run_my_cluster.sh
```

5. **Model evaluation:**
See **Model Evaluation & Interpretation** section at the end

> **Note**: Training arguments (batch size, epochs, learning rate, etc.) can be modified in the `ARGS` array within the script.


# Training
1. Create your own configuration file by copying the default template:

```bash
cp configs/default.yaml configs/my_config.yaml
```

2. Open and Edit `my_config.yaml` to specify your data path and adjust training parameters as needed:
   - Set `data.root` to point to your data folder
   - Optional: Modify hyperparameters (e.g., learning rate, batch size) based on your dataset and hardware

**Note:** Before running the following training commend lines, make sure to request GPU resources according to your HPC cluster's job scheduler (e.g., SLURM, PBS, LSF) if applicable.

> **Note:** When specifying learning rates in YAML config files, use decimal notation (e.g., `0.00001`) instead of scientific notation (e.g., `1e-5`), as some YAML parsers may incorrectly interpret scientific notation as a string.

### Single-GPU
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
    --config configs/my_config.yaml \
    --batch-size 12 \
    --num-epochs 5 \
    --temperature 0.07
```

---

## Key Parameters

### Most Commonly Tuned

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Temperature | `--temperature` | 0.07 | InfoNCE temperature. Lower = sharper distribution |
| Batch size | `--batch-size` | 12 | Per-GPU batch size |
| Epochs | `--num-epochs` | 5 | Training epochs |
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

Override Example - adjust learning rates:
```bash
torchrun --nproc_per_node=4 scripts/run_train.py \
    --config configs/default.yaml \
    --lr-img-backbone 1e-5 \
    --lr-gene-encoder 3e-4 \
    --temperature 0.1
```
> **Note:** When specifying learning rates in YAML config files, use decimal notation (e.g., `0.00001`) instead of scientific notation (e.g., `1e-5`), as some YAML parsers may incorrectly interpret scientific notation as a string.
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
| `vit_dino_v1` | ViT-S/16 DINO | Fast | **Default.** Good for most cases, fine-tuned end to end |
| `gigapath_frozen_v1` | GigaPath | Slow | Pathology-pretrained, requires HF token. Options for user who wants to dig into more morphological features. Note: due to heavy-weight of the package, the image encoder model backbone is freezed for this option |


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
  num_epochs: 5
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

## Outputs After Training

Training outputs are saved to `checkpoints/<run_name>/`:
```
checkpoints/standard_discotme_<run_name>/
├── config.yaml           # Full config (for reproducibility)
├── best_model.pth        # Best model weights
├── final_model.pth       # Final model weights
├── checkpoint_epoch5.pth # Periodic checkpoints
└── loss_history.json     # Training loss curve
```

# Model Evaluation & Interpretation
After training, you can use the following script to perform inference, generate spatial clusters, and extract the Alpha weights

1. Run Evaluation
This script will extract multi-modal embeddings from your Visium data and perform K-Means clustering across multiple scales.
```bash
python eval/run_eval_with_alpha.py \
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

# Citation

If you find DiSCoTME useful in your research, please cite:
```
Song, N., & Craig, D. (2026). Using multi-scale spatial integration of histology and transcriptomics to discover hidden tumor microenvironment motifs
```
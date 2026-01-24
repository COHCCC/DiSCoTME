## Data Preprocessing

This script prepares Visium datasets for downstream training by aligning high-resolution WSI images with spatial transcriptomics data. It crops image patches for each spot and generates corresponding gene expression vectors.

### Usage
```bash
# Example with a custom gene list
python scripts/preprocessing_usr_list.py --root /coh_labs/dits/nsong/manuscript/J13 --gene_list /path/to/gene_list.csv 

# Example using default top 2000 HVGs
python scripts/preprocessing_usr_list.py --root /coh_labs/dits/nsong/manuscript/J13
```

### Arguments

| Argument | Required | Description |
| :--- | :--- | :--- |
| `--root` | **Yes** | Path to Visium directory (must contain `.h5` file and `spatial/` folder). |
| `--gene_list` | No | CSV file with gene names in the first column (header: `Name`). If omitted, the top 2000 HVGs are used. |
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

# DiSCoTME

**Di**lated **S**patial **Co**ntrastive Learning for **T**umor **M**icro**E**nvironment Analysis

A multimodal deep learning framework that integrates spatial transcriptomics with histopathology imaging using adaptive gated fusion and dilated attention mechanisms.

## Installation
```bash
git clone https://github.com/xxx/DiSCoTME.git
cd DiSCoTME
pip install -r requirements.txt
```

## Quick Start
```bash
# 1. Copy config and change data path
cp configs/quick_start.yaml configs/my_config.yaml
# Edit my_config.yaml: set data.root to your data folder

# 2. Run training
python scripts/run_train.py --config configs/my_config.yaml
```

That's it! 🎉

---

## Training

### Single GPU
```bash
python scripts/run_train.py --config configs/default.yaml
```

### Multi-GPU (Single Node)
```bash
torchrun --nproc_per_node=4 scripts/run_train.py --config configs/default.yaml
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
| `factorized_discotme` | FactorCL variant with shared/unique decomposition |

### Image Encoders (`--image-encoder-type`)

| Name | Description |
|------|-------------|
| `gated_image_encoder` | **Default.** Adaptive gate for identity-context fusion |
| `multiscale_image_encoder` | Concat-based fusion |

### Image Backbones (`--image-backbone`)

| Name | Model | Speed | Notes |
|------|-------|-------|-------|
| `vit_dino_v1` | ViT-S/16 DINO | ⚡ Fast | **Default.** Good for most cases |
| `gigapath_frozen_v1` | GigaPath | 🐢 Slow | Pathology-pretrained, requires HF token |
| `gigapath_with_confidence` | GigaPath + confidence | 🐢 Slow | For distillation |
| `vit_dino_with_confidence` | DINO + confidence | ⚡ Fast | For distillation |

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
| `LongNet_for_spots` | 4 | 8 | Same as above |
| `LongNet_for_large_spatial` | 8 | 16 | Visium HD / large datasets |
| `LongNet_8_layers_256_dim` | 8 | 16 | Deep model |

---

## Configuration

### Minimal Config
```yaml
# configs/quick_start.yaml
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

---

## Data Format
```
data_root/
├── metadata.csv
├── tissue_positions.csv
├── patches/
│   ├── sample1_AAACAAGTATCTCCCA-1.png
│   └── ...
└── gene_expression/
    ├── sample1.h5ad
    └── ...
```

### metadata.csv
```csv
sample_id,spot_id,patch_path,gene_path
sample1,AAACAAGTATCTCCCA-1,patches/sample1_AAACAAGTATCTCCCA-1.png,gene_expression/sample1.h5ad
sample1,AAACAATCTACTAGCA-1,patches/sample1_AAACAATCTACTAGCA-1.png,gene_expression/sample1.h5ad
...
```

### tissue_positions.csv
```csv
sample_id,spot_id,x,y
sample1,AAACAAGTATCTCCCA-1,1024,2048
sample1,AAACAATCTACTAGCA-1,1124,2048
...
```

---

## SLURM Cluster

For City of Hope HPC users, see `scripts/run_ddp.sh` as a template.

Key modifications needed:
- `--partition`: Your cluster's GPU partition
- `--gres`: GPU type (e.g., `gpu:v100:4`, `gpu:a100:4`)
- Conda environment path
- Data paths
```bash
# Copy and modify for your cluster
cp scripts/run_ddp.sh scripts/my_cluster.sh
# Edit the script, then submit
sbatch scripts/slurm/my_cluster.sh
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



## License

MIT License
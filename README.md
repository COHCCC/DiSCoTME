## <font color=#00297D>Index</font>
- **Sanity check Demo data before submittion
- Demo data download: 
```bash
wget -r -np -nd ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8513nnn/GSM8513873/suppl/
wget https://huggingface.co/datasets/nina-song/SPA1_D/resolve/main/Craig_SPA1_D.tif
```
- Preprocessing (speed up this process by ```salloc -n16 -p bigmem --mem=512G```)

```bash
gunzip *.gz

for f in GSM8513873_SPA1_D_*; do
    mv "$f" "${f#GSM8513873_SPA1_D_}"
done

mkdir -p spatial
mv tissue_positions.csv scalefactors_json.json spatial/
```

- In some cases, you might need to export lib path by ```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nsong/anaconda3/envs/gigapath/lib/``` (remember to change to your envs)

```bash
python preprocessing/preprocessing_auto.py --root /coh_labs/dits/nsong/manuscript/SPA1_D
```

- Train the Model

```bash
cd TME_aware_model/
sbatch run_ddp_usr.sh   --data-root /coh_labs/dits/nsong/manuscript/SPA1_D --metadata-csv metadata.csv --tissue-positions-csv tissue_positions.csv --batch-size 12 --num-epochs 5 --learning-rate 3e-5 --weight-decay 1e-5 --temperature 0.07 --num-local 15 --num-global 0 --local-distance 400 --embed-dim 256 --proj-dim 128 --seed 42 --save-dir-base checkpoints/
```
- Eval the Model

```bash
sbatch run_visualization.sh --data-root /coh_labs/dits/nsong/manuscript/SPA1_D --metadata-csv metadata.csv --model-path /coh_labs/dits/nsong/manuscript/DiSCoTME/TME_aware_model/checkpoints/geneattn_ddp_20251207_150949/final_model.pth --output-dir /coh_labs/dits/nsong/manuscript/SPA1_D/results --output-suffix e5
```

- Final output
```
/coh_labs/dits/nsong/manuscript/SPA1_D/results/k6/fused_clusters_e5.csv
```

- [1. Introduction & Objectives](#1)
- [2. File Structure](#2)
- [3. Tasks & Workflows](#3)

## <h2 id="1"><font color=#00297D>1. Introduction & Objectives</font></h2>
## <h2 id="2"><font color=#00297D>2. Tasks & Workflows</font></h2>
## 2.1 Preprocessing
The preprocessing module has been upgraded for fully automated Visium data handling. Users only need to provide a single directory that contains:

- **H&E image** (`.tif` / `.tiff`)
- **Visium output files** (`filtered_feature_bc_matrix.h5`, `spatial/tissue_positions.csv`, etc.)

### The script will automatically:

- Detect the spot diameter from 10x metadata
- Segment the tissue and extract spot-aligned image patches
- Compute highly variable genes (HVGs)
- Generate a unified metadata file linking each spot to its coordinates, gene expression, and corresponding patch path

### Usage
Hint: 
- If on HPC such as gemini, speed up this process by ```salloc -n16 -p bigmem --mem=512G``` (dont forget to exit after using)
- Activate environment 
- In some cases, you might need to export lib path by ```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nsong/anaconda3/envs/gigapath/lib/``` (remember to change to your envs)

```bash
python preprocessing/preprocessing_auto.py --root /path/to/Visium_project
```

### Output Structure

```
CRC_08_Tumor/
├── metadata.csv                    # consolidated metadata
├── image/                          # per-spot image crops
├── genes/                          # filtered HVG expression matrix
├── ...                             # Original files
```

## 2.2 Train the Model

Train DiSCoTME using distributed data parallel (DDP, if request more than 1 node) on GPUs with contrastive learning.

### Usage
```bash
sbatch --job-name=job_name --partition=gpu-v100 --gres=gpu:2 --nodes=1 --ntasks-per-node=1 run_ddp_usr.sh \
  --data-root /path/to/Visium_or_VisiumHD_project \
  --metadata-csv metadata.csv \
  --tissue-positions-csv tissue_positions.csv \
  --batch-size 12 \
  --num-epochs 5 \
  --learning-rate 3e-5 \
  --weight-decay 1e-5 \
  --temperature 0.07 \
  --num-local 15 \
  --num-global 0 \
  --embed-dim 256 \
  --proj-dim 128 \
  --seed 42 \
  --local-distance 400 \
  --save-dir-base checkpoints/test
```

### Key Parameters

**Data Configuration:**
- `--data-root`: Path to preprocessed data directory
- `--metadata-csv`: Metadata file from preprocessing
- `--tissue-positions-csv`: Spatial coordinates file

**Training Settings:**
- `--batch-size`: Number of samples per batch (12)
- `--num-epochs`: Training epochs (5)
- `--learning-rate`: Learning rate for optimizer (3e-5)
- `--weight-decay`: L2 regularization (1e-5)

**Contrastive Learning:**
- `--temperature`: Temperature for contrastive loss (0.07)
- `--num-local`: Number of local neighbors for positive pairs (15)
- `--num-global`: Number of global samples (0 = disabled)
- `--local-distance`: Maximum distance for local neighbors in pixels (400)

**Model Architecture:**
- `--embed-dim`: Embedding dimension (256)
- `--proj-dim`: Projection head dimension (128)

**Others:**
- `--seed`: Random seed for reproducibility (42)
- `--save-dir-base`: Directory to save model checkpoints

### SLURM Configuration

- `--job-name`: Job identifier
- `--partition`: GPU partition (gpu-v100)
- `--gres`: GPU resources (2 GPUs)
- `--nodes`: Number of compute nodes (1)
- `--ntasks-per-node`: Tasks per node (1)


## 2.3 Evaluate and Visualize

Generate embeddings and visualizations for the trained model.

### Usage

```bash
sbatch run_visualization.sh \
  --data-root /path/to/Visium_project \
  --metadata-csv /path/to/Visium_project/metadata.csv \
  --model-path DiSCoTME/TME_aware_model/checkpoints/test/job_name_with_time_stamp/final_model.pth \
  --output-dir /path/to/Visium_project/results \
  --output-suffix e5
```

### Parameters

**Input Data:**
- `--data-root`: Path to the preprocessed data directory
- `--metadata-csv`: Path to the consolidated metadata file

**Model:**
- `--model-path`: Path to the trained model checkpoint (`.pth` file under TME_aware_model/save-dir-base path)

**Output:**
- `--output-dir`: Directory to save evaluation results and visualizations
- `--output-suffix`: Suffix for output files (e.g., 'e5' for epoch 5)

### Expected Outputs

The script will generate in the output directory:
- Spot embeddings extracted from the trained model (image/gene/fused embeddings)
- Kmeans clustering (k=4-10) based on spot embeddings

### Usage in Loupe browser or Seurat
The output CSV file contains barcodes and their corresponding clusters, formatted for direct integration with Seurat:

Example: ```/path/to/Visium_or_VisiumHD_project/results/k6/fused_clusters_e5.csv```
```
Barcode                    Cluster
GTCACTTCCTTCTAGA-1        1
CACGGTCTCCTTACGA-1        4
ATAGCTGCGGATAAGA-1        4
GTCAGTATGTCCGGCG-1        1
ATGTACCAGTTACTCG-1        4
ACGCTCAGTGCACCGT-1        4
```
This CSV can be directly loaded in RStudio and added to a Seurat object as metadata for downstream visualization and analysis.
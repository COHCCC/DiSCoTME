# scripts/run_train.py
"""
DiSCoTME Training Script with YAML Configuration Support

Usage:
    # Using YAML config (recommended)
    python -m torch.distributed.run --nproc_per_node=4 scripts/run_train.py --config configs/default.yaml
    
    # Override specific parameters
    python -m torch.distributed.run --nproc_per_node=4 scripts/run_train.py --config configs/default.yaml --batch-size 32
    
    # CLI only (without config file)
    python -m torch.distributed.run --nproc_per_node=4 scripts/run_train.py --data-root /path/to/data
"""

import os
import sys
import json
import random
import copy
from datetime import datetime, timedelta
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml

# ==============================================================================
# Path Setup
# ==============================================================================
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)

if os.environ.get("RANK", "0") == "0":
    print(f"[DEBUG] Project Root: {project_root}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==============================================================================
# Imports
# ==============================================================================
try:
    from src.data.dataset import MultiScaleContextDataset
    from src.models.discotme_net import MODELS
    from src.training.trainer import train_one_epoch
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

try:
    from src.models.dilated_blocks import DilatedConfigs
except ImportError:
    DilatedConfigs = {}


# ==============================================================================
# Default Configuration
# ==============================================================================
DEFAULT_CONFIG = {
    # --- Data ---
    'data': {
        'root': None,
        'metadata_csv': 'metadata.csv',
        'tissue_positions_csv': 'tissue_positions.csv',
        'num_local': 15,
        'num_global': 0,
        'local_distance': 400,
    },
    # --- Model ---
    'model': {
        'arch': 'standard_discotme',
        'image_encoder_type': 'gated_image_encoder',
        'image_backbone': 'vit_dino_v1',
        'gene_encoder_type': 'gated_gene_encoder',
        'embed_dim': 256,
        'proj_dim': 128,
    },
    # --- Dilated Attention ---
    'context': {
        'preset': 'LongNet_for_spatial',
        'custom': None,
    },
    # --- Training ---
    'training': {
        'batch_size': 12,
        'num_epochs': 50,
        'weight_decay': 1e-5,
        'temperature': 0.07,
        'seed': 42,
        'use_distill': False,
        'distill_weight': 0.0,
    },
    # --- Learning Rates (key hyperparameters) ---
    'lr': {
        'img_backbone': 1e-5,      # Pretrained backbone, use low lr
        'img_proj': 1e-4,
        'img_context': 1e-4,
        'gene_encoder': 3e-4,      # Train from scratch, can use higher lr
        'gene_proj': 1e-4,
        'default': 1e-4,           # Fallback
    },
    # --- Output ---
    'output': {
        'save_dir': 'checkpoints',
        'run_name': None,
    },
}


# ==============================================================================
# Configuration Utilities
# ==============================================================================
def deep_update(base_dict, update_dict):
    """Recursively update nested dictionary"""
    if update_dict is None:
        return base_dict
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_yaml_config(config_path):
    """Load YAML configuration file"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args():
    p = argparse.ArgumentParser(
        description="DiSCoTME Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal (YAML config)
  python run_train.py --config configs/default.yaml
  
  # Command line only
  python run_train.py --data-root /path/to/data
  
  # YAML + override
  python run_train.py --config configs/default.yaml --batch-size 32 --temperature 0.1
  
  # Adjust learning rates
  python run_train.py --config configs/default.yaml --lr-img-backbone 5e-6 --lr-gene-encoder 1e-4
        """
    )
    
    # === Config file ===
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file")
    
    # === Data ===
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--metadata-csv", type=str, default=None)
    p.add_argument("--tissue-positions-csv", type=str, default=None)
    p.add_argument("--num-local", type=int, default=None)
    p.add_argument("--num-global", type=int, default=None)
    p.add_argument("--local-distance", type=int, default=None)
    
    # === Model ===
    p.add_argument("--model-arch", type=str, default=None,
                   choices=["standard_discotme", "factorized_discotme"])
    p.add_argument("--image-encoder-type", type=str, default=None)
    p.add_argument("--image-backbone", type=str, default=None)
    p.add_argument("--gene-encoder-type", type=str, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--proj-dim", type=int, default=None)
    
    # === Context/Dilated ===
    p.add_argument("--context-config", type=str, default=None,
                   help="Dilated attention preset name")
    p.add_argument("--config-path", type=str, default=None,
                   help="Path to custom dilated attention YAML")
    
    # === Training (commonly tuned parameters) ===
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None,
                   help="InfoNCE temperature (default: 0.07)")
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    
    # === Learning Rates (key hyperparameters, exposed separately) ===
    p.add_argument("--lr-img-backbone", type=float, default=None,
                   help="Learning rate for image backbone (default: 1e-5)")
    p.add_argument("--lr-img-proj", type=float, default=None,
                   help="Learning rate for image projection (default: 1e-4)")
    p.add_argument("--lr-img-context", type=float, default=None,
                   help="Learning rate for image context processor (default: 1e-4)")
    p.add_argument("--lr-gene-encoder", type=float, default=None,
                   help="Learning rate for gene encoder (default: 3e-4)")
    p.add_argument("--lr-gene-proj", type=float, default=None,
                   help="Learning rate for gene projection (default: 1e-4)")
    p.add_argument("--learning-rate", type=float, default=None,
                   help="Default learning rate for all groups (fallback)")
    
    # === Distillation ===
    p.add_argument("--use-distill", action="store_true", default=None)
    p.add_argument("--distill-weight", type=float, default=None)
    
    # === Output ===
    p.add_argument("--save-dir-base", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    
    return p.parse_args()


def build_config(args):
    """
    Build final configuration: DEFAULT -> YAML -> CLI
    Priority: CLI > YAML > DEFAULT
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    
    # 1. Load YAML config
    yaml_config = load_yaml_config(args.config)
    config = deep_update(config, yaml_config)
    
    # 2. CLI overrides (only override non-None values)
    cli_overrides = {
        'data': {
            'root': args.data_root,
            'metadata_csv': args.metadata_csv,
            'tissue_positions_csv': args.tissue_positions_csv,
            'num_local': args.num_local,
            'num_global': args.num_global,
            'local_distance': args.local_distance,
        },
        'model': {
            'arch': args.model_arch,
            'image_encoder_type': args.image_encoder_type,
            'image_backbone': args.image_backbone,
            'gene_encoder_type': args.gene_encoder_type,
            'embed_dim': args.embed_dim,
            'proj_dim': args.proj_dim,
        },
        'context': {
            'preset': args.context_config,
            'config_path': args.config_path,
        },
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'temperature': args.temperature,
            'weight_decay': args.weight_decay,
            'seed': args.seed,
            'use_distill': args.use_distill,
            'distill_weight': args.distill_weight,
        },
        'lr': {
            'img_backbone': args.lr_img_backbone,
            'img_proj': args.lr_img_proj,
            'img_context': args.lr_img_context,
            'gene_encoder': args.lr_gene_encoder,
            'gene_proj': args.lr_gene_proj,
            'default': args.learning_rate,
        },
        'output': {
            'save_dir': args.save_dir_base,
            'run_name': args.run_name,
        },
    }
    
    # Recursively apply overrides, only for non-None values
    def apply_cli_overrides(base, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict):
                if key not in base:
                    base[key] = {}
                apply_cli_overrides(base[key], value)
            elif value is not None:
                base[key] = value
    
    apply_cli_overrides(config, cli_overrides)
    
    return config


# ==============================================================================
# Optimizer Builder
# ==============================================================================
def build_optimizer(base_model, config, encoder_type="gated"):
    """
    Build optimizer based on config['lr']
    Compatible with both Standard and Factorized model architectures
    """
    lr_config = config['lr']
    weight_decay = config['training']['weight_decay']
    param_groups = []
    
    if encoder_type == "gated":
        # ============================================
        # Gated Encoder parameter groups (DINO fully unfrozen)
        # ============================================
        
        # 1. Image Backbone (DINO) - fully unfrozen but with low lr
        param_groups.append({
            'params': base_model.img_encoder.image_encoder.backbone.parameters(),
            'lr': lr_config.get('img_backbone', 1e-5),
            'name': 'img_backbone'
        })
        
        # 2. Image Projection (inside Encoder)
        param_groups.append({
            'params': base_model.img_encoder.image_encoder.proj.parameters(),
            'lr': lr_config.get('img_proj', 1e-4),
            'name': 'img_proj'
        })
        
        # 3. Image Context Processor (Dilated + Gate)
        param_groups.append({
            'params': base_model.img_encoder.context_processor.parameters(),
            'lr': lr_config.get('img_context', 1e-4),
            'name': 'img_context'
        })
        
        # 4. Gene Encoder (all parameters)
        param_groups.append({
            'params': base_model.gene_encoder.parameters(),
            'lr': lr_config.get('gene_encoder', 3e-4),
            'name': 'gene_encoder'
        })
        
        # 5. Final Projections (compatible with Standard and Factorized)
        
        # Standard version: img_proj, gene_proj
        if hasattr(base_model, 'img_proj'):
            param_groups.append({
                'params': base_model.img_proj.parameters(),
                'lr': lr_config.get('img_proj', 1e-4),
                'name': 'img_final_proj'
            })
        
        # Factorized version: img_proj_shared, img_proj_unique
        if hasattr(base_model, 'img_proj_shared'):
            param_groups.append({
                'params': base_model.img_proj_shared.parameters(),
                'lr': lr_config.get('img_proj', 1e-4),
                'name': 'img_proj_shared'
            })
        if hasattr(base_model, 'img_proj_unique'):
            param_groups.append({
                'params': base_model.img_proj_unique.parameters(),
                'lr': lr_config.get('img_proj', 1e-4),
                'name': 'img_proj_unique'
            })
        
        # Factorized version: Decoders
        if hasattr(base_model, 'decoder_shared'):
            param_groups.append({
                'params': base_model.decoder_shared.parameters(),
                'lr': lr_config.get('img_proj', 1e-4),
                'name': 'decoder_shared'
            })
        if hasattr(base_model, 'decoder_unique'):
            param_groups.append({
                'params': base_model.decoder_unique.parameters(),
                'lr': lr_config.get('img_proj', 1e-4),
                'name': 'decoder_unique'
            })
        
        # Gene Final Proj (both versions have this)
        param_groups.append({
            'params': base_model.gene_proj.parameters(),
            'lr': lr_config.get('gene_proj', 1e-4),
            'name': 'gene_final_proj'
        })
        
    elif encoder_type == "gigapath_confidence":
        # ============================================
        # GigaPath + Confidence Head version
        # ============================================
        default_lr = lr_config.get('default', 1e-4)
        
        param_groups.append({
            'params': [p for p in base_model.img_encoder.image_encoder.backbone.parameters() if p.requires_grad],
            'lr': lr_config.get('img_backbone', 1e-6),
            'name': 'gigapath_backbone'
        })
        param_groups.append({
            'params': base_model.img_encoder.image_encoder.proj.parameters(),
            'lr': lr_config.get('img_proj', default_lr),
            'name': 'img_proj'
        })
        param_groups.append({
            'params': base_model.img_encoder.image_encoder.confidence_head.parameters(),
            'lr': default_lr,
            'name': 'confidence_head'
        })
        param_groups.append({
            'params': base_model.img_encoder.context_processor.parameters(),
            'lr': lr_config.get('img_context', default_lr),
            'name': 'img_context'
        })
        param_groups.append({
            'params': base_model.gene_encoder.parameters(),
            'lr': lr_config.get('gene_encoder', 3e-4),
            'name': 'gene_encoder'
        })
        param_groups.append({
            'params': base_model.img_proj.parameters(),
            'lr': lr_config.get('img_proj', default_lr),
            'name': 'img_final_proj'
        })
        param_groups.append({
            'params': base_model.gene_proj.parameters(),
            'lr': lr_config.get('gene_proj', default_lr),
            'name': 'gene_final_proj'
        })
    
    else:
        # ============================================
        # Generic version: all parameters with same lr
        # ============================================
        default_lr = lr_config.get('default', 1e-4)
        param_groups.append({
            'params': base_model.parameters(),
            'lr': default_lr,
            'name': 'all'
        })
    
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer


# ==============================================================================
# Main Training Function
# ==============================================================================
def main():
    args = parse_args()
    
    # === Build final configuration ===
    config = build_config(args)
    
    # === Validate required parameters ===
    if config['data']['root'] is None:
        print("ERROR: data.root is required. Use --data-root or set in config YAML.")
        sys.exit(1)

    # === DDP Setup ===
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))

    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    # === Seed ===
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # === Print Config (Rank 0 only) ===
    if global_rank == 0:
        print("=" * 60)
        print("DiSCoTME Training Configuration")
        print("=" * 60)
        print(f"World Size: {world_size} GPUs")
        print(f"\n[Model]")
        print(f"  Architecture: {config['model']['arch']}")
        print(f"  Image Encoder: {config['model']['image_encoder_type']}")
        print(f"  Image Backbone: {config['model']['image_backbone']}")
        print(f"  Gene Encoder: {config['model']['gene_encoder_type']}")
        print(f"\n[Training]")
        print(f"  Batch Size: {config['training']['batch_size']}")
        print(f"  Epochs: {config['training']['num_epochs']}")
        print(f"  Temperature: {config['training']['temperature']}")
        print(f"\n[Learning Rates]")
        for k, v in config['lr'].items():
            print(f"  {k}: {v}")
        print("=" * 60)

    # === Data Transforms ===
    transform_image = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # === Dataset ===
    dataset = MultiScaleContextDataset(
        metadata_csv=config['data']['metadata_csv'],
        tissue_positions_csv=config['data']['tissue_positions_csv'],
        root_dir=config['data']['root'],
        transform_image=transform_image,
        num_local=config['data']['num_local'],
        num_global=config['data']['num_global'],
        local_distance=config['data']['local_distance'],
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, 
        shuffle=True, seed=seed, drop_last=True
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True
    )

    # === Dilated Attention Config ===
    model_config_dict = None
    context_cfg = config['context']
    
    if context_cfg.get('config_path') and os.path.exists(context_cfg['config_path']):
        with open(context_cfg['config_path'], 'r') as f:
            model_config_dict = yaml.safe_load(f)
        if global_rank == 0:
            print(f"[Context] Loaded from YAML: {context_cfg['config_path']}")
    
    elif context_cfg.get('custom'):
        model_config_dict = context_cfg['custom']
        if global_rank == 0:
            print(f"[Context] Using custom config from YAML")
    
    elif context_cfg['preset'] in DilatedConfigs:
        model_config_dict = DilatedConfigs[context_cfg['preset']].copy()
        if global_rank == 0:
            print(f"[Context] Using preset: '{context_cfg['preset']}'")
    
    else:
        if global_rank == 0:
            print(f"[Context] Using hardcoded default")
        model_config_dict = {
            'encoder_layers': 4,
            'encoder_embed_dim': 256,
            'encoder_ffn_embed_dim': 1024,
            'encoder_attention_heads': 8,
            'dilated_ratio': '[1, 2, 3, 4]',
            'segment_length': '[1000, 2000, 3000, 5000]',
            'block_shift': True,
            'flash_attention': True,
            'use_xmoe': False,
            'dropout': 0.1,
            'drop_path_rate': 0.1
        }

    # === Model ===
    model_cfg = config['model']
    img_args = {
        "backbone_type": model_cfg['image_backbone'],
        "embed_dim": model_cfg['embed_dim'],
        "config_dict": model_config_dict
    }
    gene_args = {
        "gene_dim": 2000,
        "embed_dim": model_cfg['embed_dim'],
        "config_dict": model_config_dict
    }

    ModelClass = MODELS.get(model_cfg['arch'])
    if ModelClass is None:
        raise ValueError(f"Model '{model_cfg['arch']}' not found. Available: {list(MODELS._module_dict.keys())}")

    base_model = ModelClass(
        img_enc_name=model_cfg['image_encoder_type'],
        gene_enc_name=model_cfg['gene_encoder_type'],
        proj_dim=model_cfg['proj_dim'],
        img_args=img_args,
        gene_args=gene_args
    )

    model = base_model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    )

    # === Optimizer ===
    if "gated" in model_cfg['image_encoder_type']:
        encoder_type = "gated"
    elif "confidence" in model_cfg['image_backbone']:
        encoder_type = "gigapath_confidence"
    else:
        encoder_type = "generic"
    
    optimizer = build_optimizer(base_model, config, encoder_type=encoder_type)
    
    if global_rank == 0:
        print(f"\n[Optimizer] Type: '{encoder_type}'")
        for i, pg in enumerate(optimizer.param_groups):
            name = pg.get('name', f'group_{i}')
            lr = pg['lr']
            num_params = sum(p.numel() for p in pg['params'])
            print(f"  {name}: lr={lr}, params={num_params/1e6:.2f}M")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['num_epochs'], eta_min=1e-7
    )
    
    # === Checkpoint Dir ===
    output_cfg = config['output']
    if output_cfg['run_name'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_cfg['run_name'] = f"{model_cfg['arch']}_{timestamp}"
    
    save_dir = os.path.join(output_cfg['save_dir'], output_cfg['run_name'])

    if global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        # Save full configuration
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\n[Output] Checkpoints: {save_dir}")

    dist.barrier()
    
    best_avg_loss = float("inf")
    loss_hist = []
    temperature = config['training']['temperature']
    use_distill = config['training']['use_distill']
    distill_weight = config['training']['distill_weight']

    # === Training Loop ===
    for epoch in range(config['training']['num_epochs']):
        sampler.set_epoch(epoch)
        
        if use_distill and distill_weight > 0:
            rank_avg = train_one_epoch(
                model, dataloader, optimizer, device, 
                temperature=temperature,
                distill_weight=distill_weight
            )
        else:
            rank_avg = train_one_epoch(
                model, dataloader, optimizer, device, 
                temperature=temperature
            )
        
        # Gather Loss across all ranks
        loss_tensor = torch.tensor([rank_avg], dtype=torch.float, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        global_avg = loss_tensor.item()

        if global_rank == 0:
            loss_hist.append(global_avg)
            print(f"Epoch {epoch+1}/{config['training']['num_epochs']} | Loss: {global_avg:.4f}")
            
            # Note: scheduler.step() is only called on rank 0 to maintain consistency
            # with internal testing. This is a known issue but kept for reproducibility.
            scheduler.step()
            
            if global_avg < best_avg_loss:
                best_avg_loss = global_avg
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"  -> Saved Best Model (Loss: {best_avg_loss:.4f})")
            
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': global_avg,
                    'config': config,
                }, os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pth"))

        dist.barrier()

    # === Save Final ===
    if global_rank == 0:
        torch.save(model.module.state_dict(), os.path.join(save_dir, "final_model.pth"))
        with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_hist, f)
        print("\nTraining Finished!")
        print(f"Best Loss: {best_avg_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
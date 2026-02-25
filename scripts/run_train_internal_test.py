# scripts/run_train.py
import os, sys, json, random
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
# Path Fixes
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
    from src.models.discotme_net import MODELS  # Using Registry
    from src.training.trainer import train_one_epoch
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

try:
    from src.models.dilated_blocks import DilatedConfigs
except ImportError:
    DilatedConfigs = {}


def parse_args():
    p = argparse.ArgumentParser(description="DiSCoTME DDP training")
    
    # Data arguments
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--tissue-positions-csv", type=str, default="tissue_positions.csv")
    p.add_argument("--metadata-csv", type=str, default="metadata.csv")
    
    # Training arguments
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.07)
    
    # Dataset arguments
    p.add_argument("--num-local", type=int, default=15)
    p.add_argument("--num-global", type=int, default=0)
    p.add_argument("--local-distance", type=int, default=400)
    
    # Model arguments
    p.add_argument("--model-arch", type=str, default="standard_discotme", 
                   choices=["standard_discotme", "factorized_discotme"],
                   help="Choose model architecture: standard (original) or factorized (FactorCL)")
    p.add_argument("--image-encoder-type", type=str, default="gated_image_encoder")
    p.add_argument("--image-backbone", type=str, default="vit_dino_v1")
    p.add_argument("--gene-encoder-type", type=str, default="gated_gene_encoder")
    p.add_argument("--context-config", type=str, default="LongNet_for_spatial")
    p.add_argument("--config-path", type=str, default=None)

    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir-base", type=str, default="checkpoints")
    p.add_argument("--run-name", type=str, default=None)
    
    # [Reserved] Distillation options (Disabled by default)
    p.add_argument("--use-distill", action="store_true", help="Enable distillation loss")
    p.add_argument("--distill-weight", type=float, default=0.0)
    
    return p.parse_args()


def build_optimizer(base_model, args, encoder_type="gated"):
    """
    Build optimizer based on different encoder types.
    Compatible with both Standard and Factorized model architectures.
    """
    param_groups = []
    
    if encoder_type == "gated":
        # ============================================
        # Parameter groups for Gated Encoder (DINO fully unfrozen)
        # ============================================
        
        # 1. Image Backbone (DINO) - Fully unfrozen, but with low learning rate
        param_groups.append({
            'params': base_model.img_encoder.image_encoder.backbone.parameters(),
            'lr': 1e-5, # 1e-5 when fully unfrozen
            'name': 'img_backbone'
        })
        
        # 2. Image Projection (Internal to the Encoder)
        param_groups.append({
            'params': base_model.img_encoder.image_encoder.proj.parameters(),
            'lr': 1e-4,
            'name': 'img_proj'
        })
        
        # 3. Image Context Processor (Dilated + Gate)
        param_groups.append({
            'params': base_model.img_encoder.context_processor.parameters(),
            'lr': 1e-4,
            'name': 'img_context'
        })
        
        # 4. Gene Encoder (All parameters)
        param_groups.append({
            'params': base_model.gene_encoder.parameters(),
            'lr': 3e-4,
            'name': 'gene_encoder'
        })
        
        # 5. Final Projections (Compatible with Standard and Factorized)
        
        # Standard Version: img_proj, gene_proj
        if hasattr(base_model, 'img_proj'):
            param_groups.append({
                'params': base_model.img_proj.parameters(),
                'lr': 1e-4,
                'name': 'img_final_proj'
            })
        
        # Factorized Version: img_proj_shared, img_proj_unique
        if hasattr(base_model, 'img_proj_shared'):
            param_groups.append({
                'params': base_model.img_proj_shared.parameters(),
                'lr': 1e-4,
                'name': 'img_proj_shared'
            })
        if hasattr(base_model, 'img_proj_unique'):
            param_groups.append({
                'params': base_model.img_proj_unique.parameters(),
                'lr': 1e-4,
                'name': 'img_proj_unique'
            })
        
        # Factorized Version: Decoders
        if hasattr(base_model, 'decoder_shared'):
            param_groups.append({
                'params': base_model.decoder_shared.parameters(),
                'lr': 1e-4,
                'name': 'decoder_shared'
            })
        if hasattr(base_model, 'decoder_unique'):
            param_groups.append({
                'params': base_model.decoder_unique.parameters(),
                'lr': 1e-4,
                'name': 'decoder_unique'
            })
        
        # Gene Final Proj (Available in both versions)
        param_groups.append({
            'params': base_model.gene_proj.parameters(),
            'lr': 1e-4,
            'name': 'gene_final_proj'
        })
        
    elif encoder_type == "gigapath_confidence":
        # ============================================
        # GigaPath + Confidence Head Version (Legacy logic)
        # ============================================
        param_groups.append({
            'params': [p for p in base_model.img_encoder.image_encoder.backbone.parameters() if p.requires_grad],
            'lr': 1e-6,
            'name': 'gigapath_backbone'
        })
        param_groups.append({
            'params': base_model.img_encoder.image_encoder.proj.parameters(),
            'lr': 1e-4,
            'name': 'img_proj'
        })
        param_groups.append({
            'params': base_model.img_encoder.image_encoder.confidence_head.parameters(),
            'lr': 1e-4,
            'name': 'confidence_head'
        })
        param_groups.append({
            'params': base_model.img_encoder.context_processor.parameters(),
            'lr': 1e-4,
            'name': 'img_context'
        })
        param_groups.append({
            'params': base_model.gene_encoder.parameters(),
            'lr': 3e-4,
            'name': 'gene_encoder'
        })
        param_groups.append({
            'params': base_model.img_proj.parameters(),
            'lr': 1e-4,
            'name': 'img_final_proj'
        })
        param_groups.append({
            'params': base_model.gene_proj.parameters(),
            'lr': 1e-4,
            'name': 'gene_final_proj'
        })
    
    else:
        # ============================================
        # Generic Version: Uniform learning rate for all parameters
        # ============================================
        param_groups.append({
            'params': base_model.parameters(),
            'lr': args.learning_rate,
            'name': 'all'
        })
    
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


def main():
    args = parse_args()

    # DDP Setup
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))

    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if global_rank == 0:
        print(f"Distributed training with {world_size} GPUs. Local rank: {local_rank}")
        print(f"Model Arch: {args.model_arch}")
        print(f"Image Encoder: {args.image_encoder_type}")
        print(f"Image Backbone: {args.image_backbone}")
        print(f"Gene Encoder: {args.gene_encoder_type}")
        print(f"Use Distillation: {args.use_distill}")

    # Data Transforms
    transform_image = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Dataset
    dataset = MultiScaleContextDataset(
        metadata_csv=args.metadata_csv,
        tissue_positions_csv=args.tissue_positions_csv,
        root_dir=args.data_root,
        transform_image=transform_image,
        num_local=args.num_local,
        num_global=args.num_global,
        local_distance=args.local_distance,
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed=args.seed, drop_last=True
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True
    )

    # ==========================================================
    # Config Loading (Maintain original logic)
    # ==========================================================
    model_config_dict = None

    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            model_config_dict = yaml.safe_load(f)
        if global_rank == 0:
            print(f"[Config] Loaded from YAML: {args.config_path}")

    elif args.context_config in DilatedConfigs:
        model_config_dict = DilatedConfigs[args.context_config].copy()
        if global_rank == 0:
            print(f"[Config] Loaded preset: '{args.context_config}'")

    else:
        if global_rank == 0:
            print(f"[Config] Using hardcoded default")
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

    # Model Arguments
    img_args = {
        "backbone_type": args.image_backbone,
        "embed_dim": args.embed_dim,
        "config_dict": model_config_dict
    }
    gene_args = {
        "gene_dim": 2000,
        "embed_dim": args.embed_dim,
        "config_dict": model_config_dict
    }

    # ==========================================================
    # Model Initialization (Dynamic retrieval via Registry)
    # ==========================================================
    ModelClass = MODELS.get(args.model_arch)
    if ModelClass is None:
        raise ValueError(f"Model architecture '{args.model_arch}' not found in registry.")

    base_model = ModelClass(
        img_enc_name=args.image_encoder_type,
        gene_enc_name=args.gene_encoder_type,
        proj_dim=args.proj_dim,
        img_args=img_args,
        gene_args=gene_args
    )

    model = base_model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    )

    # ==========================================================
    # Optimizer (Auto-selection based on encoder type)
    # ==========================================================
    if "gated" in args.image_encoder_type:
        encoder_type = "gated"
    elif "confidence" in args.image_backbone:
        encoder_type = "gigapath_confidence"
    else:
        encoder_type = "generic"
    
    optimizer = build_optimizer(base_model, args, encoder_type=encoder_type)
    
    if global_rank == 0:
        print(f"[Optimizer] Using '{encoder_type}' parameter groups")
        for i, pg in enumerate(optimizer.param_groups):
            name = pg.get('name', f'group_{i}')
            lr = pg['lr']
            num_params = sum(p.numel() for p in pg['params'])
            print(f"  {name}: lr={lr}, params={num_params/1e6:.2f}M")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-7
    )
    
    # Checkpoint Dir
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.model_arch}_{timestamp}"
    
    save_dir = os.path.join(args.save_dir_base, args.run_name)

    if global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        print(f"Checkpoints will be saved to: {save_dir}")

    dist.barrier()
    
    best_avg_loss = float("inf")
    loss_hist = []

    # ==========================================================
    # Training Loop (Maintain original logic)
    # ==========================================================
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        
        # Call with different parameters depending on distillation use
        if args.use_distill and args.distill_weight > 0:
            rank_avg = train_one_epoch(
                model, dataloader, optimizer, device, 
                temperature=args.temperature,
                distill_weight=args.distill_weight
            )
        else:
            rank_avg = train_one_epoch(
                model, dataloader, optimizer, device, 
                temperature=args.temperature
            )
        
        # Gather Loss
        loss_tensor = torch.tensor([rank_avg], dtype=torch.float, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        global_avg = loss_tensor.item()

        if global_rank == 0:
            loss_hist.append(global_avg)
            print(f"Epoch {epoch+1}/{args.num_epochs} | Global Avg Loss: {global_avg:.4f}")
            scheduler.step()
            
            if global_avg < best_avg_loss:
                best_avg_loss = global_avg
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"Saved Best Model (Loss: {best_avg_loss:.4f})")
            
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': global_avg,
                }, os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pth"))

        dist.barrier()

    if global_rank == 0:
        torch.save(model.module.state_dict(), os.path.join(save_dir, "final_model.pth"))
        with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
            json.dump(loss_hist, f)
        print("Training Finished.")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
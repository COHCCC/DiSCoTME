#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
if os.environ.get("RANK", "0") == "0":
    print("[BOOT] __file__ =", __file__)
    print("[BOOT] sys.argv =", sys.argv)


# ========= path direction =========
import os, sys
from pathlib import Path

# 允许用环境变量覆盖（给少数特殊部署用）
ENV_ROOT = os.environ.get("STHISTOCLIP_ROOT")

# 默认按脚本位置推断：<repo>/TME_aware_model/<script>.py -> <repo>
guess = Path(__file__).resolve()
candidates = [Path(ENV_ROOT).resolve()] if ENV_ROOT else []
candidates += [
    guess.parents[1],        # 假设脚本在 <repo>/TME_aware_model/
    guess.parents[2],        # 备选：再往上一层（防以后改结构）
]

added = False
for root in candidates:
    if root.is_dir() and (root / "TME_aware_model").exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        added = True
        if os.environ.get("RANK", "0") == "0":
            print(f"[INFO] PROJECT_ROOT added to sys.path: {root}")
        break

if not added and os.environ.get("RANK", "0") == "0":
    print("[WARN] Could not locate project root automatically. "
          "Set STHISTOCLIP_ROOT=/path/to/repo to override.")
    
    
# ========= actual import =========
import os, json, random, time
from datetime import datetime, timedelta          # <== 增加 timedelta 方便设置超时（可选）
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ===== 你的模块 =====
from TME_aware_model.global_local_model import (
    GlobalLocalContextDataset, GlobalLocalAwareEncoder,
    ImageEncoder, ContextAwareImageEncoder, ContextAwareGeneEncoder,
    GlobalLocalMultiModalModel
)
from global_local_train import train_one_epoch, contrastive_loss


def parse_args():
    p = argparse.ArgumentParser(description="DiSCoTME DDP training (soft-coded)")
    
    # Data arguments
    p.add_argument("--data-root", type=str, required=True,
                   help="Root directory containing the data")
    p.add_argument("--tissue-positions-csv", type=str, default="tissue_positions.csv",
                   help="Path to tissue positions CSV file")  # Note: you had tissue_position_csv
    p.add_argument("--metadata-csv", type=str, default="metadata.csv",
                   help="Path to metadata CSV file")
    
    # Training arguments
    p.add_argument("--batch-size", type=int, default=12,
                   help="Batch size per GPU")
    p.add_argument("--num-epochs", type=int, default=5,
                   help="Number of training epochs")
    p.add_argument("--learning-rate", type=float, default=3e-5,
                   help="Initial learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-5,
                   help="Weight decay for AdamW optimizer")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="Temperature for contrastive loss")
    
    # Dataset arguments
    p.add_argument("--num-local", type=int, default=15,
                   help="Number of local context patches")
    p.add_argument("--num-global", type=int, default=0,
                   help="Number of global context patches")
    p.add_argument("--local-distance", type=int, default=400,
                   help="Maximum distance for local context")
    
    p.add_argument("--image-backbone", type=str, default="vit_small_patch16_224_dino",
                   help="Image encoder backbone")
    p.add_argument("--context-config", type=str, default="LongNet_for_spatial",
                   help="Context configuration name")
    # Model arguments
    p.add_argument("--embed-dim", type=int, default=256,
                   help="Embedding dimension")
    p.add_argument("--proj-dim", type=int, default=128,
                   help="Projection dimension")
    
    # Other arguments
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--save-dir-base", type=str, default="checkpoints",
                   help="Base directory for saving checkpoints")
    p.add_argument("--run-name", type=str, default=None,
                   help="Optional name for this run")
    
    return p.parse_args()
def main():
    args = parse_args()

    # ---------- 关键改动：先绑定 GPU，再 init_process_group ----------
    # LOCAL_RANK 优先，其次兼容 Slurm 的 SLURM_LOCALID
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

    # 打印可见设备，便于排错（仅 rank0 打印）
    if os.environ.get("RANK", "0") == "0":
        print(f"[DDP] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}")
        print(f"[DDP] local_rank={local_rank}")

    # 先绑定设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    else:
        raise RuntimeError("CUDA not available in this process (device_count==0). "
                           "Check srun --gres and CUDA_VISIBLE_DEVICES.")

    # 再初始化进程组（之前你是先 init 再 set_device，容易踩坑）
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=30)    # 可选：避免大作业偶发超时
        )

    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    # ---------------------------------------------------------------

    # ------- 随机种子 -------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if global_rank == 0:
        print(f"Distributed training with {world_size} GPUs.")
        print(f"Using device {device} for global_rank {global_rank} (local_rank {local_rank})")
        print(f"Effective batch size: {args.batch_size * world_size}")

    # 下面你的逻辑保持不变 ↓↓↓
    transform_image = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = GlobalLocalContextDataset(
        metadata_csv=args.metadata_csv,
        tissue_positions_csv=args.tissue_positions_csv,
        root_dir=args.data_root,
        transform_image=transform_image,
        num_local=args.num_local,
        num_global=args.num_global,
        local_distance=args.local_distance,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )

    try:
        major, minor, *_ = torch.__version__.split(".")
        pw = (int(major) > 1) or (int(major) == 1 and int(minor) >= 6)
    except Exception:
        pw = False

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=pw,
    )

    img_encoder = ContextAwareImageEncoder(
        model_name=args.image_backbone,
        pretrained=True,
        embed_dim=args.embed_dim,
        config_name=args.context_config,
    )
    gene_encoder = ContextAwareGeneEncoder(
        gene_dim=2000,
        embed_dim=args.embed_dim,
        config_name=args.context_config,
    )

    base_model = GlobalLocalMultiModalModel(
        img_encoder=img_encoder,
        gene_encoder=gene_encoder,
        proj_dim=args.proj_dim,
    )

    model = base_model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    if global_rank == 0:
        print(f"Model device check: {next(model.module.img_encoder.parameters()).device}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=(global_rank == 0)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"geneattn_ddp_{timestamp}"
    save_dir = os.path.join(args.save_dir_base, run_name)

    if global_rank == 0:
        print(f"Output directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    dist.barrier()

    best_avg_loss = float("inf")
    loss_hist = []

    if global_rank == 0:
        print(f"Starting training for {args.num_epochs} epochs...")

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        rank_avg = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
        )
        loss_tensor = torch.tensor([rank_avg], dtype=torch.float, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        global_avg = loss_tensor.item()

        if global_rank == 0:
            loss_hist.append(global_avg)
            print(f"Epoch {epoch+1}/{args.num_epochs} | Global Avg Train Loss: {global_avg:.4f}")
            scheduler.step(global_avg)
            if global_avg < best_avg_loss:
                best_avg_loss = global_avg
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_model.pth"))
                print(f"Saved new best model with avg loss: {best_avg_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                ckpt = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": global_avg,
                    "config": vars(args),
                }, ckpt)
                print(f"Saved checkpoint to {ckpt}")

        dist.barrier()

    if global_rank == 0:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(loss_hist)
            plt.title("Global Average Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            fig_path = os.path.join(save_dir, "training_loss.png")
            plt.savefig(fig_path); plt.close()
            print(f"Saved training loss plot to {fig_path}")
        except Exception as e:
            print(f"Plotting failed (ignored): {e}")

        torch.save(model.module.state_dict(), os.path.join(save_dir, "final_model.pth"))
        print(f"Training completed. Final model saved to {os.path.join(save_dir, 'final_model.pth')}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
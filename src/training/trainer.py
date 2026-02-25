# src/training/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Attempt to import distributed utilities
try:
    from .distributed_utils import gather_with_grad
except ImportError:
    from distributed_utils import gather_with_grad

# ==============================================================================
# 1. Loss Function Definitions
# ==============================================================================

def contrastive_loss(img_embs, gene_embs, temperature=0.07):
    """Standard InfoNCE Loss"""
    img_embs = F.normalize(img_embs, dim=1)
    gene_embs = F.normalize(gene_embs, dim=1)
    logits = torch.matmul(img_embs, gene_embs.transpose(0, 1)) / temperature
    labels = torch.arange(img_embs.size(0), device=img_embs.device)
    loss_i2g = F.cross_entropy(logits, labels)
    loss_g2i = F.cross_entropy(logits.transpose(0, 1), labels)
    return (loss_i2g + loss_g2i) * 0.5

def independence_loss(shared_feat, unique_feat):
    """Independence Loss (used for FactorCL)"""
    shared_norm = F.normalize(shared_feat, dim=1)
    unique_norm = F.normalize(unique_feat, dim=1)
    cos_sim = torch.sum(shared_norm * unique_norm, dim=1)
    loss_indep = torch.mean(cos_sim ** 2)
    return loss_indep

# ==============================================================================
# 2. Core Training Loop (Smart Compatibility Version)
# ==============================================================================

def train_one_epoch(
    model, 
    dataloader, 
    optimizer, 
    device="cuda", 
    temperature=0.07, 
    use_global_contrast=True,
    **kwargs
):
    model.train()
    total_loss = 0.0
    count = 0
    
    rank = dist.get_rank() if dist.is_initialized() else 0

    for batch_idx, batch in enumerate(dataloader):
        # 1. Move data to device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        # 2. Forward pass
        outputs = model(batch)
        
        # 3. Smart branch processing
        loss = None
        
        # Initialize logging variables
        log_con = 0.0
        log_rec = 0.0
        log_indep = 0.0
        current_alpha = None
        
        # ==========================================================
        # Case A: Standard Model -> 4 Outputs
        # ==========================================================
        if len(outputs) == 4:
            img_emb, gene_emb, alpha_img, alpha_gene = outputs
            
            # Fallback handling
            if alpha_img is None: alpha_img = torch.tensor(0.0)
            
            if use_global_contrast:
                img_emb_all = gather_with_grad(img_emb)
                gene_emb_all = gather_with_grad(gene_emb)
            else:
                img_emb_all, gene_emb_all = img_emb, gene_emb
            
            # Calculate standard Contrastive Loss
            loss = contrastive_loss(img_emb_all, gene_emb_all, temperature)
            
            # Record for logging
            current_alpha = alpha_img

        # ==========================================================
        # Case B: Factorized Model -> 8 Outputs
        # ==========================================================
        elif len(outputs) == 8:
            (z_img_shared, z_img_unique, z_gene_shared, 
             alpha_img, alpha_gene, 
             rec_shared, rec_unique, img_feat_raw) = outputs
            
            if use_global_contrast:
                z_img_shared_all = gather_with_grad(z_img_shared)
                z_gene_shared_all = gather_with_grad(z_gene_shared)
            else:
                z_img_shared_all, z_gene_shared_all = z_img_shared, z_gene_shared
            
            # 1. Contrastive Loss
            loss_con = contrastive_loss(z_img_shared_all, z_gene_shared_all, temperature)
            
            # 2. Recon Loss (Robust configuration tuned previously)
            rec_total = rec_shared + rec_unique
            loss_recon_total = F.mse_loss(rec_total, img_feat_raw.detach())

            # 3. Independence Loss
            loss_indep = independence_loss(z_img_shared, z_img_unique)
            
            # Total weighted loss
            loss = loss_con + 0.1 * loss_recon_total + 0.0 * loss_indep
            
            # Record for logging
            log_con = loss_con.item()
            log_rec = loss_recon_total.item()
            log_indep = loss_indep.item()
            current_alpha = alpha_img
            
        else:
            # Compatibility or error handling
            if len(outputs) == 2:
                img_emb, gene_emb = outputs
                loss = contrastive_loss(img_emb, gene_emb, temperature)
            else:
                raise ValueError(f"Unexpected output length: {len(outputs)}. Expected 4 (Standard) or 8 (Factorized).")

        # ==========================================================
        # 4. Backward Pass
        # ==========================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. Statistics
        batch_size = outputs[0].size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

        # 6. Smart Log Printing
        if batch_idx % 20 == 0 and rank == 0:
            if len(outputs) == 4:
                # Standard Logging
                log_msg = f"Batch {batch_idx}: Loss={loss.item():.4f}"
                if current_alpha is not None and isinstance(current_alpha, torch.Tensor):
                    # Handle cases where alpha might be a scalar or tensor
                    val = current_alpha.mean().item() if current_alpha.numel() > 1 else current_alpha.item()
                    log_msg += f" | Alpha={val:.3f}"
            
            elif len(outputs) == 8:
                # Factorized Logging
                log_msg = (f"Batch {batch_idx}: Total={loss.item():.4f} | "
                           f"Con={log_con:.3f} Rec={log_rec:.3f} Indep={log_indep:.3f}")
                if current_alpha is not None and isinstance(current_alpha, torch.Tensor):
                    val = current_alpha.mean().item() if current_alpha.numel() > 1 else current_alpha.item()
                    log_msg += f" | Alpha={val:.3f}"
            else:
                log_msg = f"Batch {batch_idx}: Loss={loss.item():.4f}"
            
            print(log_msg)

    return total_loss / count if count > 0 else 0
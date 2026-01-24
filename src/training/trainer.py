# src/training/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# 尝试导入分布式工具
try:
    from .distributed_utils import gather_with_grad
except ImportError:
    from distributed_utils import gather_with_grad

# ==============================================================================
# 1. 损失函数定义 (保留所有工具)
# ==============================================================================

def contrastive_loss(img_embs, gene_embs, temperature=0.07):
    """标准 InfoNCE 损失"""
    img_embs = F.normalize(img_embs, dim=1)
    gene_embs = F.normalize(gene_embs, dim=1)
    logits = torch.matmul(img_embs, gene_embs.transpose(0, 1)) / temperature
    labels = torch.arange(img_embs.size(0), device=img_embs.device)
    loss_i2g = F.cross_entropy(logits, labels)
    loss_g2i = F.cross_entropy(logits.transpose(0, 1), labels)
    return (loss_i2g + loss_g2i) * 0.5

def independence_loss(shared_feat, unique_feat):
    """[保留] 独立性损失 (供 FactorCL 使用)"""
    shared_norm = F.normalize(shared_feat, dim=1)
    unique_norm = F.normalize(unique_feat, dim=1)
    cos_sim = torch.sum(shared_norm * unique_norm, dim=1)
    loss_indep = torch.mean(cos_sim ** 2)
    return loss_indep

# ==============================================================================
# 2. 核心训练循环 (智能兼容版)
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
        # 1. 数据搬运
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        # 2. 前向传播
        outputs = model(batch)
        
        # 3. 智能分支处理
        loss = None
        
        # 日志变量初始化
        log_con = 0.0
        log_rec = 0.0
        log_indep = 0.0
        current_alpha = None
        
        # ==========================================================
        # Case A: Standard Model (你现在的版本) -> 4 Outputs
        # ==========================================================
        if len(outputs) == 4:
            img_emb, gene_emb, alpha_img, alpha_gene = outputs
            
            # 容错处理
            if alpha_img is None: alpha_img = torch.tensor(0.0)
            
            if use_global_contrast:
                img_emb_all = gather_with_grad(img_emb)
                gene_emb_all = gather_with_grad(gene_emb)
            else:
                img_emb_all, gene_emb_all = img_emb, gene_emb
            
            # 只算 Contrastive Loss
            loss = contrastive_loss(img_emb_all, gene_emb_all, temperature)
            
            # 记录用于日志
            current_alpha = alpha_img

        # ==========================================================
        # Case B: Factorized Model (备用) -> 8 Outputs
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
            
            # 1. Con Loss
            loss_con = contrastive_loss(z_img_shared_all, z_gene_shared_all, temperature)
            
            # 2. Recon Loss (我们之前调好的稳健配置)
            rec_total = rec_shared + rec_unique
            loss_recon_total = F.mse_loss(rec_total, img_feat_raw.detach())

            # 3. Indep Loss
            loss_indep = independence_loss(z_img_shared, z_img_unique)
            
            # 备用权重 (Con + 0.1 Rec + 0.0 Indep)
            loss = loss_con + 0.1 * loss_recon_total + 0.0 * loss_indep
            
            # 记录用于日志
            log_con = loss_con.item()
            log_rec = loss_recon_total.item()
            log_indep = loss_indep.item()
            current_alpha = alpha_img
            
        else:
            # 兼容旧代码或报错
            if len(outputs) == 2:
                img_emb, gene_emb = outputs
                loss = contrastive_loss(img_emb, gene_emb, temperature)
            else:
                raise ValueError(f"Unexpected output length: {len(outputs)}. Expected 4 (Standard) or 8 (Factorized).")

        # ==========================================================
        # 4. 反向传播
        # ==========================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. 统计
        batch_size = outputs[0].size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

        # 6. 智能日志打印
        if batch_idx % 20 == 0 and rank == 0:
            if len(outputs) == 4:
                # Standard 日志
                log_msg = f"Batch {batch_idx}: Loss={loss.item():.4f}"
                if current_alpha is not None and isinstance(current_alpha, torch.Tensor):
                     # 处理 alpha 可能为 scalar 或 tensor 的情况
                    val = current_alpha.mean().item() if current_alpha.numel() > 1 else current_alpha.item()
                    log_msg += f" | Alpha={val:.3f}"
            
            elif len(outputs) == 8:
                # Factorized 日志
                log_msg = (f"Batch {batch_idx}: Total={loss.item():.4f} | "
                           f"Con={log_con:.3f} Rec={log_rec:.3f} Indep={log_indep:.3f}")
                if current_alpha is not None and isinstance(current_alpha, torch.Tensor):
                    val = current_alpha.mean().item() if current_alpha.numel() > 1 else current_alpha.item()
                    log_msg += f" | Alpha={val:.3f}"
            else:
                log_msg = f"Batch {batch_idx}: Loss={loss.item():.4f}"
            
            print(log_msg)

    return total_loss / count if count > 0 else 0


# # 尝试导入分布式工具 (根据你的文件结构调整)
# try:
#     from .distributed_utils import gather_with_grad
# except ImportError:
#     # 如果是在同一目录下运行，作为兜底
#     from distributed_utils import gather_with_grad

# # ==============================================================================
# # 1. 损失函数定义
# # ==============================================================================

# def contrastive_loss(img_embs, gene_embs, temperature=0.07):
#     """
#     标准 InfoNCE 损失 (Global Contrastive Loss)
#     """
#     # 归一化，确保在超球面上
#     img_embs = F.normalize(img_embs, dim=1)
#     gene_embs = F.normalize(gene_embs, dim=1)
    
#     # 计算相似度矩阵: [Batch_Global, Batch_Global]
#     logits = torch.matmul(img_embs, gene_embs.transpose(0, 1)) / temperature
    
#     # 标签是对角线 (因为 img_i 和 gene_i 是一对)
#     labels = torch.arange(img_embs.size(0), device=img_embs.device)
    
#     # 双向 CrossEntropy
#     loss_i2g = F.cross_entropy(logits, labels)
#     loss_g2i = F.cross_entropy(logits.transpose(0, 1), labels)
    
#     return (loss_i2g + loss_g2i) * 0.5


# def uncertainty_distillation_loss(f_final_fused, f_local_image, confidence):
#     teacher_target = f_local_image.detach()
#     mse_dist = torch.mean((f_final_fused - teacher_target) ** 2, dim=1, keepdim=True)
    
#     # 原始加权损失
#     weighted_loss = (confidence * mse_dist).mean()
    
#     # [NEW] 正则化：惩罚 confidence 太低
#     # 鼓励 confidence 均值接近 0.5
#     conf_reg = ((confidence.mean() - 0.5) ** 2) * 0.5
    
#     return weighted_loss + conf_reg

# def independence_loss(shared_feat, unique_feat):
#     """
#     [FactorCL 新增] 独立性损失
#     强迫 shared 和 unique 特征正交（互不相关），避免信息混淆。
#     """
#     # 归一化
#     shared_norm = F.normalize(shared_feat, dim=1)
#     unique_norm = F.normalize(unique_feat, dim=1)
    
#     # 计算余弦相似度
#     # 我们希望相似度接近 0 (即正交)
#     cos_sim = torch.sum(shared_norm * unique_norm, dim=1)
    
#     # 最小化相似度的平方
#     loss_indep = torch.mean(cos_sim ** 2)
#     return loss_indep
# # ==============================================================================
# # 2. 核心训练循环
# # ==============================================================================

# def train_one_epoch(
#     model, 
#     dataloader, 
#     optimizer, 
#     device="cuda", 
#     temperature=0.07, 
#     use_global_contrast=True,
#     **kwargs # 接收额外参数防止报错
# ):
#     model.train()
#     total_loss = 0.0
#     count = 0
    
#     rank = dist.get_rank() if dist.is_initialized() else 0

#     for batch_idx, batch in enumerate(dataloader):
#         # 1. 数据搬运
#         batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
#         # 2. 前向传播
#         outputs = model(batch)
        
#         # # ==========================================================
#         # # 3. 智能解包 (student_teacher 设计的核心)
#         # # ==========================================================
#         # 我们检查返回值的数量来判断是否使用了 "Confidence" 版本的 Encoder
#         has_distill_info = False
        
#         if isinstance(outputs, tuple) and len(outputs) == 5:
#             # 命中了新逻辑！
#             # img_emb: 最终融合投影后的特征 (Student)
#             # gene_emb: 基因特征
#             # img_conf: 置信度 (B, 1)
#             # img_raw:  GigaPath 原始 1536 维特征 (本次暂未使用，留作备用)
#             # img_feat_local: GigaPath 经过 proj 后的局部特征 (Teacher)
#             img_emb, gene_emb, img_conf, img_raw, img_feat_local = outputs
#             has_distill_info = True
#         else:
#             # 只是普通的旧逻辑
#             img_emb, gene_emb = outputs
#             has_distill_info = False
#         # ==========================================================
#         # 3. 结果解包 (适配双侧门控字典返回)
#         # ==========================================================
#         # ==========================================================
#         # Case A: Standard Model (返回 4 个值) -> 走老路
#         # ==========================================================
#         if len(outputs) == 4:
#             img_emb, gene_emb = outputs[0], outputs[1]
#             alpha_img = outputs[2]
#             alpha_gene = outputs[3]
            
#             if use_global_contrast:
#                 img_emb_all = gather_with_grad(img_emb)
#                 gene_emb_all = gather_with_grad(gene_emb)
#             else:
#                 img_emb_all, gene_emb_all = img_emb, gene_emb
            
#             loss = contrastive_loss(img_emb_all, gene_emb_all, temperature)

#         # ==========================================================
#         # Case B: Factorized Model (返回 8 个值) -> 走 Residual Disentanglement 新路
#         # ==========================================================
#         elif len(outputs) == 8:
#             # 解包：注意顺序要和 discotme_net.py 里 return 的一致
#             (z_img_shared, z_img_unique, z_gene_shared, 
#              alpha_img, alpha_gene, 
#              rec_shared, rec_unique, img_feat_raw) = outputs
            
#             # --- 1. Contrastive Loss (Shared <-> Gene) ---
#             # 只有 Shared 部分参与基因对齐
#             if use_global_contrast:
#                 z_img_shared_all = gather_with_grad(z_img_shared)
#                 z_gene_shared_all = gather_with_grad(z_gene_shared)
#             else:
#                 z_img_shared_all, z_gene_shared_all = z_img_shared, z_gene_shared
            
#         # ... Contrastive Loss (保持不变) ...
#             loss_con = contrastive_loss(z_img_shared_all, z_gene_shared_all, temperature)
            
#             # --- 2. Reconstruction Loss (修改版) ---
            
#             # [逻辑] 我们不要求 Shared 自己还原图片，
#             # 我们要求 (Shared + Unique) 合力还原图片。
#             # 就像拼图：Shared 拼好了肿瘤部分，Unique 负责把剩下的出血部分拼上去。
            
#             # rec_total = rec_shared + rec_unique
            
#             # img_feat_raw 是原始 DINO 特征 (Teacher)
#             # 1. 强迫 Shared 长得像人 (视觉锚点)
#             rec_total = rec_shared + rec_unique
#             loss_recon_total = F.mse_loss(rec_total, img_feat_raw.detach())

#             # 2. Indep 保持关闭 (这是为了救 z_img_shared 的画质)
#             loss_indep = independence_loss(z_img_shared, z_img_unique)
            
#             # --- Total Loss ---
#             # 关键修改：Recon 权重降回 0.1 (之前的 0.5+1.0 太大了)
#             # 关键修改：Indep 权重设为 0.0 (允许 Shared 学视觉特征)
            
#             loss = loss_con + 0.1 * loss_recon_total + 0.0 * loss_indep
            
#         else:
#             raise ValueError(f"Unexpected output length: {len(outputs)}")
#         # ==========================================================
#         # 5. 计算 Distillation Loss (支线任务)
#         # ==========================================================
#         # loss_distill = torch.tensor(0.0, device=device)
        
#         # if has_distill_info and distill_weight > 0:
#         #     # 注意：img_emb 是经过了 img_proj 的，img_feat_local 还没过 img_proj
#         #     # 这是一个细节点！我们需要让 Teacher 和 Student 在同一个空间对比。
            
#         #     # Student: img_emb (已经是投影后的 128/256 维)
#         #     # Teacher: img_feat_local (Encoder出来的 256 维) -> 还需要过一次 img_proj 吗？
#         #     # 让我们看 discotme_net.py: 
#         #     #   img_emb = self.img_proj(img_feat_context_aware)
#         #     # 所以 Teacher 也应该通过 img_proj 才能在同一维度对比。
            
#         #     # 获取共享的 projector (兼容 DDP)
#         #     if hasattr(model, 'module'):
#         #         projector = model.module.img_proj
#         #     else:
#         #         projector = model.img_proj
            
#         #     # Teacher 前向 (不传梯度)
#         #     with torch.no_grad():
#         #         teacher_proj = projector(img_feat_local)
                
#         #     loss_distill = uncertainty_distillation_loss(
#         #         f_final_fused=img_emb, 
#         #         f_local_image=teacher_proj, 
#         #         confidence=img_conf
#         #     )

#         # loss = loss_con + distill_weight * loss_distill
        
#         # ==========================================================
#         # 6. 反向传播 (原版）
#         # ==========================================================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # 7. 统计与日志
#         batch_size = img_emb.size(0)
#         total_loss += loss.item() * batch_size
#         count += batch_size

#         # 仅在 Rank 0 打印调试信息
#         if batch_idx % 20 == 0 and rank == 0:
#             if len(outputs) == 4:
#                 # Case A: Standard Model
#                 log_msg = f"Batch {batch_idx}: Loss={loss.item():.4f}"
#                 if alpha_img is not None:
#                     log_msg += f" | Alpha_Img={alpha_img.mean().item():.3f}"
#                 if alpha_gene is not None:
#                     log_msg += f" Alpha_Gene={alpha_gene.mean().item():.3f}"
                    
#             elif len(outputs) == 8:
#                 # Case B: Factorized Model
#                 log_msg = f"Batch {batch_idx}: Total={loss.item():.4f}"
#                 log_msg += f" | Con={loss_con.item():.3f} Recon={loss_recon_total.item():.3f}"
#                 log_msg += f" Indep={loss_indep.item():.3f}"
                
#                 if alpha_img is not None:
#                     log_msg += f" | Alpha={alpha_img.mean().item():.3f}"
            
#             print(log_msg)

#     return total_loss / count if count > 0 else 0
        
        # ==========================================================
        # 6. 反向传播 (适应factorcl）)
        # ==========================================================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Logging
#         batch_size = batch['target_img'].size(0)
#         total_loss += loss.item() * batch_size
#         count += batch_size
        
# # Logging
#         if batch_idx % 20 == 0 and rank == 0:
#             # [修正] 打印 Shared Recon 和 Unique Recon 分项
#             log_msg = (f"Batch {batch_idx}: Total={loss.item():.4f} | "
#                        f"Con={loss_con.item():.3f} "
#                        f"Indep={loss_indep.item():.3f}")         # 打印独立性 (应该是0)
            
#             if alpha_img is not None:
#                 # 如果是 Tensor 就取 mean，防止报错
#                 a_i = alpha_img.mean().item() if isinstance(alpha_img, torch.Tensor) else 0.0
#                 log_msg += f" | Alpha={a_i:.3f}"
            
#             print(log_msg)

#     return total_loss / count if count > 0 else 0
    
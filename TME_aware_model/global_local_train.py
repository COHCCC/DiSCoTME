# train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

#################################################
# 训练组件
#################################################
def contrastive_loss(img_embs, gene_embs, temperature=0.07):
    """标准InfoNCE损失函数"""
    # 归一化
    img_embs = F.normalize(img_embs, dim=1)
    gene_embs = F.normalize(gene_embs, dim=1)
    
    # 计算相似度矩阵
    logits = torch.matmul(img_embs, gene_embs.transpose(0, 1)) / temperature  # [batch_size, batch_size]
    
    # 标签是对角线索引 (正样本对)
    labels = torch.arange(img_embs.size(0), device=img_embs.device)
    
    # 计算交叉熵损失
    loss_i2g = F.cross_entropy(logits, labels)
    loss_g2i = F.cross_entropy(logits.transpose(0, 1), labels)
    
    return (loss_i2g + loss_g2i) * 0.5


def train_one_epoch(model, dataloader, optimizer, device="cuda", temperature=0.07):
    model.train()
    #scaler = GradScaler()
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(dataloader):
        # --- 在第一批的时候输出一下信息即可 ---
        if batch_idx == 0:
            print("Model is on:", next(model.parameters()).device)
            # 这里最好用字典遍历把 batch 转到 GPU 再检查
        
        # 这里正确地把 batch 中的张量转到 device 上
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        ## f16 to support a100
        # batch = {k: (v.to(device).half() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # if batch_idx % 10 == 0:
        #     # 假设 local_len 和 global_len 已经在 __getitem__ 中返回了
        #     local_lens = batch.get("local_len", None)
        #     global_lens = batch.get("global_len", None)
        #     if local_lens is not None and global_lens is not None:
        #         print(f"Batch {batch_idx}: local lengths = {local_lens}, global lengths = {global_lens}", flush=True)
        
        # 前向传播
        img_emb, gene_emb = model(batch)
        loss = contrastive_loss(img_emb, gene_emb, temperature)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        # optimizer.zero_grad()

        # with autocast():
        #     img_emb, gene_emb = model(batch)
        #     loss = contrastive_loss(img_emb, gene_emb, temperature)

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # 记录损失
        batch_size = batch['target_img'].size(0)
        total_loss += loss.item() * batch_size
        count += batch_size

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / count if count > 0 else 0


# # train.py
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler

# #################################################
# # 训练组件
# #################################################
# def contrastive_loss(img_embs, gene_embs, temperature=0.07):
#     """标准InfoNCE损失函数"""
#     # 归一化
#     img_embs = F.normalize(img_embs, dim=1)
#     gene_embs = F.normalize(gene_embs, dim=1)
    
#     # 计算相似度矩阵
#     logits = torch.matmul(img_embs, gene_embs.transpose(0, 1)) / temperature  # [batch_size, batch_size]
    
#     # 标签是对角线索引 (正样本对)
#     labels = torch.arange(img_embs.size(0), device=img_embs.device)
    
#     # 计算交叉熵损失
#     loss_i2g = F.cross_entropy(logits, labels)
#     loss_g2i = F.cross_entropy(logits.transpose(0, 1), labels)
    
#     return (loss_i2g + loss_g2i) * 0.5


# def train_one_epoch(model, dataloader, optimizer, device="cuda", temperature=0.07):
#     model.train()
#     # scaler = GradScaler()
#     total_loss = 0.0
#     count = 0

#     for batch_idx, batch in enumerate(dataloader):
#         if batch_idx == 0:
#             print("Model is on:", next(model.parameters()).device)

#         batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
#         # batch = {k: (v.to(device, dtype=torch.float16) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
#         img_emb, gene_emb = model(batch)
#         loss = contrastive_loss(img_emb, gene_emb, temperature)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
#         # optimizer.zero_grad()

#         # with autocast():
#         #     img_emb, gene_emb = model(batch)
#         #     loss = contrastive_loss(img_emb, gene_emb, temperature)

#         # scaler.scale(loss).backward()
#         # scaler.step(optimizer)
#         # scaler.update()

#         batch_size = batch['target_img'].size(0)
#         total_loss += loss.item() * batch_size
#         count += batch_size

#         if (batch_idx + 1) % 10 == 0:
#             print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

#     return total_loss / count if count > 0 else 0
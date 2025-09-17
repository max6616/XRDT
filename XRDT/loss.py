import torch
import torch.nn as nn
import torch_scatter


class CombinedLoss(nn.Module):
    """
    计算组合损失。
    [OPTIMIZED VERSION] 针对 num_hypo == 1 的情况增加了高效的向量化计算路径。
    """
    def __init__(self, miller_weight=1.0, lattice_weight=1.0, sg_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.miller_weight = miller_weight
        self.lattice_weight = lattice_weight
        self.sg_weight = sg_weight

        # Miller损失函数: Focal Loss
        # 晶格参数回归损失: MSE
        self.lattice_loss_fn = nn.MSELoss()
        # 单标签模式无需多假设padding
    
    def focal_loss(self, inputs, targets, alpha=1.0, gamma=2.0, reduction='mean'):
        """ Focal loss function. """
        # 使用 label_smoothing 和 reduction='none' 计算基础的交叉熵损失
        ce_loss = nn.functional.cross_entropy(inputs, targets, label_smoothing=0.1, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        if reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

    def forward(self, predictions, miller_targets, crystal_targets, offsets):
        device = predictions['h'].device
        # 期望 miller_targets 为 [N, 3]
        targets_h = miller_targets[:, 0]
        targets_k = miller_targets[:, 1]
        targets_l = miller_targets[:, 2]

        # Miller 分类损失
        loss_h = self.focal_loss(predictions['h'], targets_h)
        loss_k = self.focal_loss(predictions['k'], targets_k)
        loss_l = self.focal_loss(predictions['l'], targets_l)
        miller_loss = loss_h + loss_k + loss_l
        loss_h_item = loss_h.item()
        loss_k_item = loss_k.item()
        loss_l_item = loss_l.item()

        # --- 2. & 3. 晶格和空间群损失 (这部分逻辑对于两种路径是通用的) ---
        pred_lattice = predictions['lattice_params']
        target_lattice = crystal_targets['lattice']
        valid_crystal_mask = (crystal_targets['space_group'].view(-1) != -1)
        
        lattice_loss = torch.tensor(0.0, device=device)
        if valid_crystal_mask.sum() > 0 and self.lattice_weight > 0:
            # 只对有有效标签的样本计算损失
            lattice_loss = self.lattice_loss_fn(pred_lattice[valid_crystal_mask], target_lattice[valid_crystal_mask])
            # 防止损失值过大，使用张量方式避免同步
            lattice_loss = torch.clamp(lattice_loss, max=torch.tensor(0.1, device=device, dtype=lattice_loss.dtype))

        sg_loss = torch.tensor(0.0, device=device)
        if self.sg_weight > 0 and valid_crystal_mask.sum() > 0:
            pred_sg = predictions['space_group'][valid_crystal_mask]
            target_sg = crystal_targets['space_group'].view(-1)[valid_crystal_mask].to(torch.long)
            sg_loss = self.focal_loss(pred_sg, target_sg, reduction='mean')

        # --- 4. 加权总损失 ---
        total_loss = (self.miller_weight * miller_loss +
                      self.lattice_weight * lattice_loss +
                      self.sg_weight * sg_loss)

        return {
            'total_loss': total_loss,
            'loss_miller': miller_loss,
            'loss_h': torch.tensor(loss_h_item, device=device),
            'loss_k': torch.tensor(loss_k_item, device=device),
            'loss_l': torch.tensor(loss_l_item, device=device),
            'loss_lattice': lattice_loss,
            'loss_sg': sg_loss,
        }

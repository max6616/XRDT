import torch
import torch.nn as nn
import torch_scatter

class CombinedLoss(nn.Module):
    def __init__(self, miller_weight=1.0, lattice_weight=1.0, sg_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.miller_weight = miller_weight
        self.lattice_weight = lattice_weight
        self.sg_weight = sg_weight
        self.lattice_loss_fn = nn.SmoothL1Loss(beta=0.1)

    def ce_loss(self, inputs, targets, reduction='mean', label_smoothing=0.0):
        ce = nn.functional.cross_entropy(inputs, targets, label_smoothing=label_smoothing, reduction='none')
        if reduction == 'mean':
            return ce.mean()
        return ce

    def forward(self, predictions, miller_targets, crystal_targets, offsets, point_loss_mask=None, clip_lattice_loss=False):
        device = predictions['h'].device
        targets_h = miller_targets[:, 0]
        targets_k = miller_targets[:, 1]
        targets_l = miller_targets[:, 2]

        if point_loss_mask is not None:
            mask = point_loss_mask.to(device=device, dtype=torch.bool)
            loss_h_vec = self.ce_loss(predictions['h'], targets_h, reduction='none')
            loss_k_vec = self.ce_loss(predictions['k'], targets_k, reduction='none')
            loss_l_vec = self.ce_loss(predictions['l'], targets_l, reduction='none')

            def masked_mean(vec, m):
                if torch.any(m):
                    return vec[m].mean()
                return torch.tensor(0.0, device=device, dtype=vec.dtype)

            loss_h = masked_mean(loss_h_vec, mask)
            loss_k = masked_mean(loss_k_vec, mask)
            loss_l = masked_mean(loss_l_vec, mask)
        else:
            loss_h = self.ce_loss(predictions['h'], targets_h)
            loss_k = self.ce_loss(predictions['k'], targets_k)
            loss_l = self.ce_loss(predictions['l'], targets_l)
        miller_loss = loss_h + loss_k + loss_l
        loss_h_item = loss_h.item()
        loss_k_item = loss_k.item()
        loss_l_item = loss_l.item()

        pred_lattice = predictions['lattice_params']
        target_lattice = crystal_targets['lattice']
        valid_crystal_mask = (crystal_targets['space_group'].view(-1) != -1)
        
        lattice_loss = torch.tensor(0.0, device=device)
        if valid_crystal_mask.sum() > 0 and self.lattice_weight > 0:
            lattice_loss = self.lattice_loss_fn(pred_lattice[valid_crystal_mask], target_lattice[valid_crystal_mask])
            if clip_lattice_loss:
                lattice_loss = torch.clamp(lattice_loss, max=torch.tensor(1.0, device=device, dtype=lattice_loss.dtype))

        sg_loss = torch.tensor(0.0, device=device)
        if self.sg_weight > 0 and valid_crystal_mask.sum() > 0:
            pred_sg = predictions['space_group'][valid_crystal_mask]
            target_sg = crystal_targets['space_group'].view(-1)[valid_crystal_mask].to(torch.long)
            sg_loss = self.ce_loss(pred_sg, target_sg, reduction='mean')

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


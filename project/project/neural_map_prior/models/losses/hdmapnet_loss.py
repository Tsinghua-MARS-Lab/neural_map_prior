import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import LOSSES


@LOSSES.register_module()
class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight, loss_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_weight = loss_weight

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss * self.loss_weight


@LOSSES.register_module()
class DiscriminativeLoss(nn.Module):
    def __init__(self, embed_dim, delta_v, delta_d, loss_weight_cfg):
        super(DiscriminativeLoss, self).__init__()
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

        self.var_weight = loss_weight_cfg['var']
        self.dist_weight = loss_weight_cfg['dist']
        self.reg_weight = loss_weight_cfg['reg']

    def forward(self, embedding, seg_gt):
        if embedding is None:
            return 0, 0, 0
        bs = embedding.shape[0]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(bs):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(F.relu(
                    torch.norm(embedding_i - mean_i.reshape(self.embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d) ** 2) / (num_lanes * (num_lanes - 1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = (var_loss / bs) * self.var_weight
        dist_loss = (dist_loss / bs) * self.dist_weight
        reg_loss = (reg_loss / bs) * self.reg_weight
        return var_loss, dist_loss, reg_loss


@LOSSES.register_module()
class DirectionLoss(torch.nn.Module):
    def __init__(self, loss_weight):
        super(DirectionLoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss(reduction='none')
        self.loss_weight = loss_weight

    def forward(self, direction, direction_mask):
        lane_mask = (1 - direction_mask[:, 0]).unsqueeze(1)
        ytgt = direction_mask

        ypred = torch.softmax(direction, 1)
        loss = self.loss_fn(ypred, ytgt)

        loss = (loss * lane_mask).sum() / (lane_mask.sum() * loss.shape[1] + 1e-6)

        return loss * self.loss_weight

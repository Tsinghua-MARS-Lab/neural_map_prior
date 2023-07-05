import torch

from mmdet3d.models.builder import build_loss
from ..hdmapnet_utils.angle_diff import calc_angle_diff


class HdmapNetLosses(torch.nn.Module):
    def __init__(self, cfg):
        super(HdmapNetLosses, self).__init__()

        self.dir_loss = build_loss(cfg['loss_dir'])
        self.embed_loss = build_loss(cfg['loss_embed'])
        self.seg_loss = build_loss(cfg['loss_seg'])
        self.angle_class = cfg['angle_class']
        self.direction_pred = cfg['direction_pred']

    def forward(self, gts: dict(), preds: dict):

        seg_loss = self.seg_loss(preds['preds_map'], gts['seg_map'].argmax(dim=1))
        embed_loss = self.embed_loss(preds['embedded'], gts['inst_mask'].sum(1))

        losses_dict = dict(
            seg_loss=seg_loss,
            embed_var_loss=embed_loss[0],
            embed_dist_loss=embed_loss[1],
            embed_reg_loss=embed_loss[2],
        )

        if self.direction_pred:
            direction_loss = self.dir_loss(
                preds['direction'], gts['direction_mask'])
            losses_dict.update(dict(direction_loss=direction_loss, ))

        loss = 0
        for name, var in losses_dict.items():
            loss = loss + var

        log_vars = {k: v.item() for k, v in losses_dict.items()}

        if self.direction_pred:
            angle_diff = calc_angle_diff(
                preds['direction'], gts['direction_mask'], self.angle_class)
            log_vars.update(dict(angle_diff=angle_diff))

        return loss, log_vars

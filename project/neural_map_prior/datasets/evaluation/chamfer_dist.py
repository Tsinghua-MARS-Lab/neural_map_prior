import torch
import torch.nn.functional as F
# load from https://github.com/krrish94/chamferdist
# which is borrowed from point3D
from chamferdist.chamfer import knn_points

from .utils import get_pos_idx


class ChamferDistance(torch.nn.Module):

    def __init__(self, H, W, resH, resW, device, normalize=False):
        super(ChamferDistance, self).__init__()

        if int(H) == 100:
            hs = torch.linspace(0, H, steps=int(H // resH + 1))
            ws = torch.linspace(0, W, steps=int(W // resW + 1))
        else:
            hs = torch.linspace(0, H, steps=int(H // resH))
            ws = torch.linspace(0, W, steps=int(W // resW))
        hh, ww = torch.meshgrid(hs, ws)

        self.canvas = torch.stack([hh, ww], -1)  # H, W, 2
        self.canvas = self.canvas.to(device).reshape(-1, 2)
        self.inf = max(H, W)
        self.normalize = normalize

    @classmethod
    def _chamfer_distance(
            cls,
            source_cloud: torch.Tensor,
            target_cloud: torch.Tensor,
            source_mask: torch.Tensor,
            target_mask: torch.Tensor,
    ):
        '''
            source_cloud: batchsize_source, lengths_source, dim_source
            target_cloud: batchsize_target, lengths_target, dim_target

        '''

        batchsize_source, lengths_source, dim_source = source_cloud.shape
        batchsize_target, lengths_target, dim_target = target_cloud.shape

        chamfer_dist = None

        source_nn = knn_points(
            source_cloud,
            target_cloud,
            lengths1=source_mask,
            lengths2=target_mask,
            K=1,
        )

        target_nn = knn_points(
            target_cloud,
            source_cloud,
            lengths1=target_mask,
            lengths2=source_mask,
            K=1,
        )

        # Forward Chamfer distance (batchsize_source, lengths_source)
        chamfer_forward = source_nn.dists[..., 0].sqrt()
        # Backward Chamfer distance (batchsize_source, lengths_target)
        chamfer_backward = target_nn.dists[..., 0].sqrt()

        return chamfer_forward, chamfer_backward

    def forward(self, *args, **kwargs):
        return self._chamfer_distance(*args, **kwargs)

    def semantic_mask_chamfer_dist_cum_parallel(self, pred, tgt, threshold=5.):
        '''
            Args:
                pred B, C, H, W
                target: B, C, H, W

        '''

        B, C, H, W = pred.shape

        pred_flat = pred.reshape(-1, H * W)
        tgt_flat = tgt.reshape(-1, H * W)

        pred_mask = pred_flat > 0
        tgt_mask = tgt_flat > 0

        # We need take out the idx from pred and tgt which
        # satisfy following conditions.
        # none of them have positive in this idx then
        # we set the distance with threshold
        thresh_pred_mask = (~(tgt_mask.sum(-1, keepdim=True) > 0)) & pred_mask
        thresh_tgt_mask = (~(pred_mask.sum(-1, keepdim=True) > 0)) & tgt_mask

        # Step 1 get rid of empty mask for fast forward.
        pred_idx, pred_mask = get_pos_idx(pred_mask)
        pred_pos = self.canvas[pred_idx]  # N, npos, 2
        pred_pos = pred_mask.unsqueeze(-1) * pred_pos + \
                   (~pred_mask.unsqueeze(-1)) * self.inf
        thresh_pred_mask = thresh_pred_mask.gather(-1, pred_idx)

        tgt_idx, tgt_mask = get_pos_idx(tgt_mask)
        tgt_pos = self.canvas[tgt_idx]  # N, npos, 2
        tgt_pos = tgt_mask.unsqueeze(-1) * tgt_pos + \
                  (~tgt_mask.unsqueeze(-1)) * self.inf
        thresh_tgt_mask = thresh_tgt_mask.gather(-1, tgt_idx)

        # Step 2 calculate the distance and clamp the abonormal data
        CD_pred, CD_label = self._chamfer_distance(
            pred_pos, tgt_pos, pred_mask.sum(-1), tgt_mask.sum(-1))
        CD_pred[thresh_pred_mask] = threshold
        CD_label[thresh_tgt_mask] = threshold

        # Normalization
        CD_pred = (CD_pred * pred_mask).sum(-1) / \
                  pred_mask.sum(-1).masked_fill(pred_mask.sum(-1) == 0, 1)
        CD_label = (CD_label * tgt_mask).sum(-1) / \
                   tgt_mask.sum(-1).masked_fill(tgt_mask.sum(-1) == 0, 1)

        return (
            CD_pred.reshape(B, C).sum(0),
            CD_label.reshape(B, C).sum(0),
            (pred_mask.sum(-1) > 0).reshape(B, C).sum(0),
            (tgt_mask.sum(-1) > 0).reshape(B, C).sum(0),)

    @classmethod
    def instance_cd(cls, cd_pred, cd_label, threshold=5.):
        '''
            Args:
                pred List[]
                pred_mask: List[]

                target: B, C, M, seq_len_gt, 2
                tgt_mask: B, C, M, seq_len_gt

                C is the class number
            Returns:
                CD_pred: C
                CD_label: C

        '''

        cd_pred, pred_matched_label = cd_pred.sum(-1).min(-1)
        cd_label, label_matched_pred = cd_label.sum(-1).min(-2)

        return cd_pred, cd_label

    @classmethod
    def vec_cd_dist(cls, pred, pred_mask, tgt, tgt_mask):
        '''
            Args:
                pred: B, N, interpolate_len, 2
                pred_mask: N
                target: B, M, interpolate_len, 2
                tgt_mask: B, M

            Returns:
                CD_pred: B x N x M x len
                CD_label: B x N x M x len
                valid_mask: B x N x M

        '''

        device = pred.device

        B, N, inter_size, _ = pred.shape
        B, M, inter_size, _ = tgt.shape

        pred = pred[:, :, None].repeat(1, 1, M, 1, 1)
        tgt = tgt[:, None].repeat(1, N, 1, 1, 1)

        # Note here we need to determine each combination is valid.
        # valid_mask: B, N, M
        valid_mask = pred_mask[:, :, None] & tgt_mask[:, None]

        pred_mask = pred_mask[:, :, None].repeat(1, 1, M)
        tgt_mask = tgt_mask[:, None].repeat(1, N, 1)

        # flatten
        pred = pred.view(B * N * M, inter_size, 2)
        tgt = tgt.view(B * N * M, inter_size, 2)

        valid_mask = valid_mask.view(-1)
        pred_mask = pred_mask.view(-1)
        tgt_mask = tgt_mask.view(-1)

        # Calculate the distance and clamp the abonormal data
        seq_length = torch.full(pred_mask.shape, inter_size, device=device)
        pred_length = seq_length * pred_mask
        tgt_length = seq_length * tgt_mask

        # B*N*M, len
        CD_pred, CD_label = \
            cls._chamfer_distance(
                pred, tgt, pred_length, tgt_length)

        # B*N*M, len
        _valid_mask = valid_mask[:, None].repeat(1, inter_size)
        CD_pred = \
            torch.where(_valid_mask, CD_pred,
                        torch.tensor(20000., device=device))
        CD_label = \
            torch.where(_valid_mask, CD_label,
                        torch.tensor(20000., device=device))

        # reshape to original shape
        CD_pred = CD_pred.view(B, N, M, inter_size)
        CD_label = CD_label.view(B, N, M, inter_size)
        valid_mask = valid_mask.view(B, N, M)

        return CD_pred, CD_label, valid_mask


def get_match_idx(cd_pred, cd_label, valid_mask):
    # '''
    #     Args:
    #         cd_pred: B, N, M, len
    #         cd_label: B, N, M, len
    #         valid_mask: B, N, M

    #     Returns:
    #         cd_pred_

    # '''

    # cd_pred = cd_pred.sum(-1)
    # cd_label = cd_label.sum(-1)

    # cd_pred
    pass


def upsampler(line, pts=100, align_corners=True):
    line = F.interpolate(line.transpose(-1, -2), size=pts,
                         mode='linear', align_corners=align_corners)
    line = line.transpose(-1, -2)

    return line


if __name__ == '__main__':
    import torch

    line1 = torch.tensor([
        [1, 4], [3, 5], [5, 5], [7, 6]
    ])

    line2 = torch.tensor([
        [1, 3], [3, 4], [5, 4], [7, 5]
    ])

    line3 = torch.tensor([
        [4, 1], [3, 3], [5, 3], [7, 3]
    ])

    gt = torch.stack((line2, line3), dim=0).type(torch.float32)
    pred = line1[None].type(torch.float32)

    # import ipdb; ipdb.set_trace()
    # with mmcv.Timer():
    #     gt = upsampler(gt, pts=10)
    #     pred = upsampler(pred, pts=10)

    import matplotlib.pyplot as plt

    for i in gt:
        i = i.numpy()
        plt.plot(i[:, 0], i[:, 1], 'o', color='red')
        plt.plot(i[:, 0], i[:, 1], '-', color='red')

    for i in pred:
        i = i.numpy()
        plt.plot(i[:, 0], i[:, 1], 'o', color='blue')
        plt.plot(i[:, 0], i[:, 1], '-', color='blue')

    plt.savefig('test.png')

    tgt = gt[None]  # B,C,M,seq_len,2
    tgt_mask = torch.ones_like(tgt[..., 0, 0]).type(torch.bool)  # B,C,M,seq_len,2

    pred = pred.repeat(3, 1, 1)
    pred = pred[None]
    pred_mask = torch.ones_like(pred[..., 0, 0]).type(
        torch.bool)  # B,C,N
    pred_mask[:, 1:] = False

    CD1, CD2, valid_maks = \
        ChamferDistance.vec_cd_dist(pred, pred_mask, tgt, tgt_mask)

    print(CD1.mean(-1)[0])
    print(CD2.mean(-1)[0])
    print(valid_maks[0])

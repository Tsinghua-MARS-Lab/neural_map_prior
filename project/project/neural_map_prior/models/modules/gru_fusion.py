import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.spatial.transform import Rotation as R


def gen_matrix(ego2global_rotation, ego2global_translation):
    rotation_xyz = np.roll(ego2global_rotation, shift=-1)
    trans = np.eye(4)
    trans[:3, 3] = ego2global_translation
    trans[:3, :3] = R.from_quat(rotation_xyz).as_matrix()
    return trans


def gen_ego2ego_matrix(ori_ego2global_rotation,
                       ref_ego2global_rotation,
                       ori_ego2global_translation,
                       ref_ego2global_translation):
    ori_trans = gen_matrix(ori_ego2global_rotation, ori_ego2global_translation)
    ref_trans = gen_matrix(ref_ego2global_rotation, ref_ego2global_translation)
    return np.linalg.inv(ori_trans) @ ref_trans


def get_sample_coords(bev_bound_w, bev_bound_h, bev_w, bev_h):
    '''
    Args:
        bev_bound_w (tuple:2):
        bev_bound_h (tuple:2):
        bev_w (int):
        bev_h (int):

    Returns: (bev_h, bev_w, 4)

    '''
    sample_coords = torch.stack(torch.meshgrid(
        torch.linspace(bev_bound_w[0], bev_bound_w[1], int(bev_w), dtype=torch.float32),
        torch.linspace(bev_bound_h[0], bev_bound_h[1], int(bev_h), dtype=torch.float32)
    ), axis=2).transpose(1, 0)
    zeros = torch.zeros((sample_coords.shape[0], sample_coords.shape[1], 1), dtype=sample_coords.dtype)
    ones = torch.ones((sample_coords.shape[0], sample_coords.shape[1], 1), dtype=sample_coords.dtype)
    sample_coords = torch.cat([sample_coords, zeros, ones], dim=-1)
    return sample_coords


def get_coords_resample(bev_feature, pad_bev_feature, ego2ego, real_h=30, real_w=60):
    '''
    Args:
        bev_feature (B, bev_h, bev_w, C):
        pad_bev_feature (B, bev_h, bev_w, C):
        ego2ego (4, 4):
        real_h (int):
        real_w (int):
    Returns: (B, bev_h, bev_w, C)

    '''
    device = bev_feature.device
    B, bev_h, bev_w, C = bev_feature.size()

    bev_bound_h, bev_bound_w = \
        [(-row[0] / 2 + row[0] / row[1] / 2, row[0] / 2 - row[0] / row[1] / 2)
         for row in ((real_h, bev_h), (real_w, bev_w))]
    grid_len_h = real_h / bev_h
    grid_len_w = real_w / bev_w

    bev_coords = get_sample_coords(bev_bound_w, bev_bound_h, bev_w, bev_h).to(device)
    ego2ego = bev_coords.new_tensor(ego2ego)

    bev_coords = bev_coords.reshape(-1, 4).permute(1, 0)
    trans_bev_coords = ego2ego @ bev_coords

    bev_coord_w = trans_bev_coords[0, :]
    bev_coord_h = trans_bev_coords[1, :]
    bev_coord_mask = \
        (bev_bound_w[0] <= bev_coord_w) & (bev_coord_w < bev_bound_w[1]) & \
        (bev_bound_h[0] <= bev_coord_h) & (bev_coord_h < bev_bound_h[1])

    bev_index_w = torch.floor((bev_coord_w - bev_bound_w[0]) / grid_len_w).to(torch.int64)
    bev_index_h = torch.floor((bev_coord_h - bev_bound_h[0]) / grid_len_h).to(torch.int64)

    bev_coord_mask = bev_coord_mask.reshape(bev_h, bev_w)
    bev_index_w = bev_index_w.reshape(bev_h, bev_w)
    bev_index_h = bev_index_h.reshape(bev_h, bev_w)

    index_h, index_w = torch.where(bev_coord_mask.reshape(bev_h, bev_w))
    overlap_feats = bev_feature[:, index_h, index_w, :]
    pad_bev_feature[:, bev_index_h[index_h, index_w], bev_index_w[index_h, index_w], :] += overlap_feats


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=256, bev_h=100, bev_w=200):
        super(ConvGRU, self).__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.batch_size = 1

    def forward(self, h, x, hist_index_h=None, hist_index_c=None, **kwargs):
        # h = h.permute(0, 3, 1, 2)
        # x = x.permute(0, 3, 1, 2)

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        x = torch.cat([r * h, x], dim=1)
        q = torch.tanh(self.convq(x))

        h = (1 - z) * h + z * q
        # return h.permute(0, 2, 3, 1)
        return h


class ConvGRUDeformable(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=256, bev_h=100, bev_w=200):
        super(ConvGRUDeformable, self).__init__()
        kernel_size = 3
        padding = kernel_size // 2
        self.padding = padding
        self.convz_offset = nn.Conv2d(hidden_dim + input_dim, 2 * kernel_size * kernel_size,
                                      kernel_size=kernel_size, padding=padding, bias=True)
        self.convr_offset = nn.Conv2d(hidden_dim + input_dim, 2 * kernel_size * kernel_size,
                                      kernel_size=kernel_size, padding=padding, bias=True)
        self.convq_offset = nn.Conv2d(hidden_dim + input_dim, 2 * kernel_size * kernel_size,
                                      kernel_size=kernel_size, padding=padding, bias=True)
        nn.init.constant_(self.convz_offset.weight, 0.)
        nn.init.constant_(self.convz_offset.bias, 0.)
        nn.init.constant_(self.convr_offset.weight, 0.)
        nn.init.constant_(self.convr_offset.bias, 0.)
        nn.init.constant_(self.convq_offset.weight, 0.)
        nn.init.constant_(self.convq_offset.bias, 0.)

        self.regular_convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=kernel_size, padding=padding,
                                       bias=False)
        self.regular_convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=kernel_size, padding=padding,
                                       bias=False)
        self.regular_convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size=kernel_size, padding=padding,
                                       bias=False)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.batch_size = 1

    def forward(self, h, x, hist_index_h=None, hist_index_c=None, **kwargs):
        # h = h.permute(0, 3, 1, 2)
        # x = x.permute(0, 3, 1, 2)

        hx = torch.cat([h, x], dim=1)
        offset_z = self.convz_offset(hx)
        z = torch.sigmoid(torchvision.ops.deform_conv2d(input=hx,
                                                        offset=offset_z,
                                                        weight=self.regular_convz.weight,
                                                        bias=self.regular_convz.bias,
                                                        padding=(self.padding, self.padding)))
        offset_r = self.convr_offset(hx)
        r = torch.sigmoid(torchvision.ops.deform_conv2d(input=hx,
                                                        offset=offset_r,
                                                        weight=self.regular_convr.weight,
                                                        bias=self.regular_convr.bias,
                                                        padding=(self.padding, self.padding)))
        x = torch.cat([r * h, x], dim=1)
        offset_q = self.convq_offset(x)
        q = torch.tanh(torchvision.ops.deform_conv2d(input=x,
                                                     offset=offset_q,
                                                     weight=self.regular_convq.weight,
                                                     bias=self.regular_convq.bias,
                                                     padding=(self.padding, self.padding)))

        h = (1 - z) * h + z * q
        # return h.permute(0, 2, 3, 1)
        return h


if __name__ == '__main__':
    convgru = ConvGRU()
    h = torch.randn(1, 100 * 200, 256)
    x = torch.randn(1, 100 * 200, 256)
    output = convgru(h, x)
    print(output.shape)

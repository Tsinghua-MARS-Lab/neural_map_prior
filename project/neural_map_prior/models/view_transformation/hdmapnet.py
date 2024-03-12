import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from mmdet3d.models.builder import NECKS

from .homography import IPM


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C
        
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320 + 112, self.C)
    
    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()
        
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x
        
        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x
        
        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x
    
    def forward(self, x):
        return self.get_eff_depth(x)


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)
    
    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H * W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(
                B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs


@NECKS.register_module()
class HDMapNetBackbone(nn.Module):
    def __init__(self,
                 outC,
                 img_res,
                 camC=64,
                 xbound=None,
                 ybound=None,
                 zbound=None,
                 lidar=False,
                 lidar_dim=128,
                 lidar_only=False,
                 up_sample_scale=2):
        super(HDMapNetBackbone, self).__init__()
        
        self.camC = camC
        self.downsample = 16
        
        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        final_H, final_W = nx[1].item(), nx[0].item()
        
        self.camencode = CamEncode(camC)
        fv_size = (img_res[0] // self.downsample, img_res[1] // self.downsample)
        bv_size = (final_H // 5, final_W // 5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)
        
        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4 * res_x / final_W]
        ipm_ybound = [-res_x / 2, res_x / 2, 2 * res_x / final_H]
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=camC, extrinsic=True)
        self.up_sampler = nn.Upsample(
            scale_factor=up_sample_scale, mode='bilinear', align_corners=True)
        
        self.lidar = lidar
        self.lidar_only = lidar_only
        if lidar:
            self.pp = PointPillarEncoder(lidar_dim, xbound, ybound, zbound)
    
    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        
        Rs = torch.eye(4, device=rots.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts
        
        post_RTs = None
        
        return Ks, RTs, post_RTs
    
    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH // self.downsample, imW // self.downsample)
        return x
    
    def forward(self, imgs, img_metas, points=None):
        
        if self.lidar_only:
            ptensor, pmask = points
            lidar_feature = self.pp(ptensor, pmask)
            return lidar_feature
        
        x = self.get_cam_feats(imgs)
        x = self.view_fusion(x)
        
        device = x.device
        
        # prepare batched image metas
        desired_keys = ['cam_intrinsics', 'ego2global_translation',
                        'cam2ego_rotations', 'cam2ego_translations', 'post_rots', 'post_trans', 'yaw_pitch_roll']
        batched_keys = ['cam_intrinsics', 'ego2global_translation',
                        'cam2ego_rotations', 'cam2ego_translations', 'yaw_pitch_roll']
        batched_image_matas = dict(
            [(k, []) for k in img_metas[0].keys() if k in desired_keys])
        
        for img_meta in img_metas:
            for k in desired_keys:
                batched_image_matas[k].append(img_meta[k])
        
        for k in batched_image_matas.keys():
            if k in batched_keys:
                batched_image_matas[k] = torch.tensor(batched_image_matas[k]).to(device)
        
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(
            batched_image_matas['cam_intrinsics'],
            batched_image_matas['cam2ego_rotations'],
            batched_image_matas['cam2ego_translations'],
            batched_image_matas['post_rots'],
            batched_image_matas['post_trans'])
        # from camera system to BEV
        topdown = self.ipm(
            x, Ks, RTs, batched_image_matas['ego2global_translation'], batched_image_matas['yaw_pitch_roll'], post_RTs)
        topdown = self.up_sampler(topdown)
        
        if self.lidar:
            ptensor, pmask = points
            lidar_feature = self.pp(ptensor, pmask)
            topdown = torch.cat([topdown, lidar_feature], dim=1)
        
        return topdown

import torch
from mmdet3d.models.builder import NECKS
from mmdet3d.models.builder import build_neck
from torch import nn


@NECKS.register_module()
class LSS(nn.Module):
    def __init__(self, embed_dims, transformer):
        super(LSS, self).__init__()
        self.transformer = build_neck(transformer)
    
    def get_intri_and_extri(self, img_metas):
        cam2ego_rotations = []
        cam2ego_translations = []
        cam_intrinsics = []
        for img_meta in img_metas:
            cam2ego_rotations.append(img_meta['cam2ego_rotations'])
            cam2ego_translations.append(img_meta['cam2ego_translations'])
            cam_intrinsics.append(img_meta['cam_intrinsics'])
        cam2ego_rotations = torch.tensor(cam2ego_rotations)
        cam2ego_translations = torch.tensor(cam2ego_translations)
        cam_intrinsics = torch.tensor(cam_intrinsics)
        return cam_intrinsics, cam2ego_rotations, cam2ego_translations
    
    def forward(self, imgs, imgs_feats, **kwargs):
        """

        Args:
            imgs (batch_size, num_cams, height, width):
            imgs_feats (List): len = 4
                torch.Size([1, 6, 256, 116, 200]) 1 / 8
                torch.Size([1, 6, 256, 58, 100]) 1 / 16
                torch.Size([1, 6, 256, 29, 50]) 1 / 32
                torch.Size([1, 6, 256, 15, 25]) 1 / 64
            **kwargs ():

        Returns:
            # bev_feats(tensor): batch_size, bev_h * bev_w, embed_dims
            bev_feats(tensor): batch_size, bev_h, bev_w, embed_dims

        """
        
        img_metas = kwargs['img_metas']
        # torch.Size([1, 6, 256, 58, 100])
        cam_feats = imgs_feats[1]
        batch_size, N, C, H, W = cam_feats.size()
        cam_feats = cam_feats.reshape(batch_size * N, C, H, W)
        
        resize = imgs.shape[-2] / 900
        # assert resize == imgs.shape[-1] / 1600
        device = imgs.device
        intrins, rots, trans = self.get_intri_and_extri(img_metas)
        rots = rots.unsqueeze(0).to(device).to(imgs.dtype)
        trans = trans.unsqueeze(0).to(device).to(imgs.dtype)
        intrins = intrins.unsqueeze(0).to(device).to(imgs.dtype)
        
        post_rot2 = torch.eye(2) * resize
        post_rots = torch.eye(3).to(device).to(imgs.dtype)
        post_rots[:2, :2] = post_rot2
        post_rots = post_rots.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, N, 1, 1)
        post_trans = torch.zeros(3).to(device).to(imgs.dtype)
        post_trans = post_trans.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, N, 1)
        cam_feats = cam_feats.unsqueeze(0).unsqueeze(0)
        bev_feats = self.transformer([cam_feats, rots, trans, intrins, post_rots, post_trans])
        
        # bev_feats torch.Size([1, 20000, 256])
        bev_feats = bev_feats.squeeze(1)
        bev_feats = bev_feats.permute(0, 2, 3, 1)
        bev_feats = bev_feats.flatten(1, 2)
        return bev_feats

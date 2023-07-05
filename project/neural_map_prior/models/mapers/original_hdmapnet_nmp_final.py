import torch
from torch import nn

from mmdet3d.models.builder import build_backbone, build_head, build_neck, MAPPERS
from project.neural_map_prior.map_tiles.nuscenes_map_slice import get_bev_centerness, gen_matrix
from project.neural_map_prior.models.mapers.base_mapper import BaseMapper
from project.neural_map_prior.models.mapers.loss_utils import HdmapNetLosses
from project.neural_map_prior.models.mapers.map_global_memory import MapGlobalMemory
from project.neural_map_prior.models.modules.gru_fusion import ConvGRU


@MAPPERS.register_module()
class OriginalHDMapNet(BaseMapper):

    def __init__(self,
                 map_bev_attrs,
                 dist_cfg,
                 img_backbone,
                 img_neck,
                 view_transformation_cfg,
                 head_cfg,
                 loss_cfg,
                 map_attribute=None,
                 open_nmp=False,
                 **kwargs):
        super(OriginalHDMapNet, self).__init__()

        self.map_bev_attrs = map_bev_attrs
        self.bev_h = map_bev_attrs['bev_h_']
        self.bev_w = map_bev_attrs['bev_w_']
        self.real_h = map_bev_attrs['real_h_']
        self.real_w = map_bev_attrs['real_w_']
        self.embed_dims = view_transformation_cfg['embed_dims']
        self.open_nmp = open_nmp

        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)

        self.view_transformation = build_neck(view_transformation_cfg)
        self.head = build_head(head_cfg)
        self.loss = HdmapNetLosses(loss_cfg)

        if dist_cfg:
            self.img_backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.img_backbone)
            self.img_neck = nn.SyncBatchNorm.convert_sync_batchnorm(self.img_neck)
            self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

        if self.open_nmp:
            self.epoch = -1
            self.single_gpu = map_attribute['single_gpu']
            self.bev_attribute = \
                {'bev_h': self.bev_h, 'bev_w': self.bev_w,
                 'real_h': self.real_h, 'real_w': self.real_w}
            self.gm = MapGlobalMemory(map_attribute, self.bev_attribute)
            self.conv_gru_net = ConvGRU(
                hidden_dim=self.embed_dims, input_dim=self.embed_dims, bev_h=self.bev_h, bev_w=self.bev_w)
            self.convz = nn.Conv2d(
                self.embed_dims + self.embed_dims + 3, self.embed_dims, kernel_size=3, padding=1, bias=False)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def extract_img_feat(self, imgs):
        """
        Args:
            imgs (batch_size, num_cams, height, width):
            torch.Size([1, 6, 3, 928, 1600])

        Returns:
            list() len = 4
            torch.Size([1, 6, 256, 116, 200]) 1 / 8
            torch.Size([1, 6, 256, 58, 100]) 1 / 16
            torch.Size([1, 6, 256, 29, 50]) 1 / 32
            torch.Size([1, 6, 256, 15, 25]) 1 / 64
        """
        bs, num_cams, channel, height, width = imgs.size()
        imgs = imgs.reshape(bs * num_cams, channel, height, width)
        img_feats = self.img_backbone(imgs)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            bn, channel, height, width = img_feat.size()
            img_feat = img_feat.reshape(bs, num_cams, channel, height, width)
            img_feats_reshaped.append(img_feat)
        return img_feats_reshaped

    def flatte_feat_and_transpose(self, feats):
        batch_size, _, channel = feats.size()
        feats = feats.reshape(batch_size, self.bev_h, self.bev_w, channel)
        return feats.permute(0, 3, 1, 2)

    def update_func(self, bev_queries, bev_feats):
        bev_centerness = get_bev_centerness(
            bev_h=self.bev_h, bev_w=self.bev_w,
            real_h=self.real_h, real_w=self.real_w
        ).to(bev_feats.device).unsqueeze(0).permute(0, 3, 1, 2)

        z = torch.sigmoid(self.convz(torch.cat(
            [bev_queries, bev_feats, bev_centerness], dim=1)))
        bev_queries = (1 - z) + bev_queries
        bev_feats = z + bev_feats
        return bev_queries, bev_feats

    def reshape2bev(self, batch_size, feat):
        return feat.reshape(batch_size, self.bev_h, self.bev_w, self.embed_dims)

    def forward_single(self, imgs, **kwargs):
        batch_size = imgs.size(0)
        img_metas = kwargs['img_metas']

        imgs_feats = self.extract_img_feat(imgs)

        if self.open_nmp:
            # define prior_bev
            prior_bev = torch.zeros(
                (batch_size, self.bev_h, self.bev_w, self.embed_dims)).to(imgs.device)

            # get prior_bev from global memory
            for ib in range(prior_bev.shape[0]):
                token = img_metas[ib]['sample_idx']
                img_meta = img_metas[ib]
                trans = gen_matrix(
                    img_meta['ego2global_rotation'],
                    img_meta['ego2global_translation']
                )
                if self.training:
                    self.gm.take_map_prior(prior_bev[ib:ib + 1], token, img_meta, 'train', trans)
                else:
                    self.gm.take_map_prior(prior_bev[ib:ib + 1], token, img_meta, 'val', trans)

            prior_bev = prior_bev.flatten(1, 2)
            img_metas[0]['global_hist_state'] = prior_bev

        bev_feats = self.view_transformation(imgs, imgs_feats, **kwargs)


        if self.open_nmp:
            cur_bev = self.reshape2bev(batch_size, bev_feats)
            prior_bev = self.reshape2bev(batch_size, prior_bev)

            # transpose cur_bev and prior_bev
            cur_bev = cur_bev.permute(0, 3, 1, 2)
            prior_bev = prior_bev.permute(0, 3, 1, 2)
            
            if self.use_centerness:
                prior_bev, cur_bev = self.update_func(prior_bev, cur_bev)

            # update cur_bev and prior_bev
            new_hist_state = self.conv_gru_net(prior_bev, cur_bev)

            # transpose new_hist_state
            new_hist_state = new_hist_state.permute(0, 2, 3, 1)

            # replace prior_bev in global memory
            for ib in range(new_hist_state.shape[0]):
                token = img_metas[ib]['sample_idx']
                img_meta = img_metas[ib]
                trans = gen_matrix(
                    img_meta['ego2global_rotation'],
                    img_meta['ego2global_translation']
                )
                if self.training:
                    self.gm.replace_map_prior(new_hist_state[ib:ib + 1], token, img_meta, 'train', trans)
                else:
                    self.gm.replace_map_prior(new_hist_state[ib:ib + 1], token, img_meta, 'val', trans)

            # transpose new_hist_state
            bev_feats = new_hist_state.flatten(1, 2)

        # bev_feats torch.Size([1, 20000, 256])
        bev_feats = self.flatte_feat_and_transpose(bev_feats)
        bev_feature = list([bev_feats])
        preds_dict = self.head(bev_feature)
        return preds_dict

    def reset_map(self, dataset):
        gpu_id = 0 if self.single_gpu else torch.distributed.get_rank()
        self.gm.reset_define_map(
            self.epoch,
            gpu_id=gpu_id,
            dataset=dataset,
            map_slices_name=[
                'map_slice_float_dict',
                'map_slice_int_dict'
            ],
            map_attribute_func={
                'map_slice_int_dict':
                    self.gm.gen_map_slice_int
            }
        )

    def forward_train(self, **kwargs):
        """
        Args:
            **kwargs ():
            # kwargs.keys()
            # dict_keys(['img_metas', 'img', 'rasterized_gt', 'vectors'])
            # kwargs['img'].shape
            # torch.Size([1, 6, 3, 928, 1600])
            # kwargs['rasterized_gt'].keys()
            # dict_keys(['seg_map', 'inst_mask', 'direction_mask'])
            # kwargs['rasterized_gt']['seg_map'].shape
            # torch.Size([1, 4, 200, 400])
            # kwargs['rasterized_gt']['inst_mask'].shape
            # torch.Size([1, 4, 200, 400])
            # kwargs['rasterized_gt']['direction_mask'].shape
            # torch.Size([1, 37, 200, 400])
            # kwargs['img_metas'][0].keys()
            # dict_keys(
            #     ['img_filenames', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape',
            #      'cam_intrinsics', 'img_norm_cfg',
            #      'sample_idx', 'cam2ego_rotations',
            #      'cam2ego_translations', 'ego2global_translation', 'ego2global_rotation',
            #      'ego2img', 'yaw_pitch_roll', 'post_trans', 'post_rots'])
            # kwargs['vectors'][0] 10
            # kwargs['vectors'][0][0] 3
            # kwargs['vectors'][0][0][0] 100, 3

        Returns:
            # preds_dict['preds_map'].shape torch.Size([1, 4, 200, 400])
            # preds_dict['embedded'].shape torch.Size([1, 16, 200, 400])
            # preds_dict['direction'].shape torch.Size([1, 37, 200, 400])

        """

        imgs = kwargs['img']
        rasterized_gt = kwargs['rasterized_gt']

        if self.open_nmp:
            dataset = 'train'
            self.reset_map(dataset)

        preds_dict = self.forward_single(imgs, **kwargs)
        loss, log_vars = self.loss(rasterized_gt, preds_dict)
        return loss, log_vars, imgs.size()[0]

    def forward_test(self, **kwargs):
        imgs = kwargs['img']
        img_metas = kwargs['img_metas']

        if self.open_nmp:
            dataset = 'val'
            self.reset_map(dataset)

        preds_dict = self.forward_single(imgs, **kwargs)
        token = [img_meta['sample_idx'] for img_meta in img_metas]
        return self.head.post_process(preds_dict, kwargs['vectors'], token)

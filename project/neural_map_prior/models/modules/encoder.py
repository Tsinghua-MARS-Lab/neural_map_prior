import torch
from mmcv.cnn.bricks.registry import \
    (TRANSFORMER_LAYER,
     TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import auto_fp16
from mmcv.utils import ext_loader

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .utils import get_reference_points, point_sampling

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    def __init__(self,
                 *args,
                 pc_range=None,
                 num_points_in_pillar=4,
                 return_intermediate=False,
                 dataset_type='nuscenes',
                 use_no_back_proj=True,
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate

        self.fp16_enabled = False
        self.use_no_back_proj = use_no_back_proj

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        '''
        Args:
            bev_query (bs, num_query, embed_dims):
            key (num_cams, num_feat_flatten, feat_dims):
            value (same as key):
            *args ():
            bev_h (int):
            bev_w (int):
            bev_pos (bs, num_query, embed_dims):
            spatial_shapes (num_levels, 2):
            level_start_index (num_levels):
            valid_ratios (None):
            prev_bev ():
            shift (2):
            **kwargs ():

        Returns:

        '''

        intermediate = []
        if not isinstance(bev_query, (list, tuple)):
            batch_size = bev_query.size(0)
            device = bev_query.device
            dtype = bev_query.dtype
            output = bev_query
            num_bev_queue = 1
        else:
            batch_size = bev_query[0].size(0)
            device = bev_query[0].device
            dtype = bev_query[0].dtype
            output = bev_query[0]
            num_bev_queue = len(bev_query)

        ref_2d = get_reference_points(
            bev_h,
            bev_w,
            dim='2d',
            bs=batch_size,
            device=device,
            dtype=dtype
        )
        ref_3d = get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim='3d',
            bs=batch_size,
            device=device,
            dtype=dtype
        )
        reference_points_cam, bev_mask = \
            point_sampling(
                ref_3d,
                self.pc_range,
                kwargs['img_metas']
            )

        hybird_ref_2d = torch.cat([ref_2d, ref_2d], 0)
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                num_bev_queue=num_bev_queue,
                **kwargs)

            bev_query = output

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        if isinstance(output, (list, tuple)):
            return output[0]
        else:
            return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(
                     type='ReLU',
                     inplace=True),
                 norm_cfg=dict(
                     type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)

        self.fp16_enabled = False
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(
        #    ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                bev_mask=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                num_bev_queue=2,
                **kwargs):
        '''

        Args:
            query (bs, num_query, embed_dims):
            key (num_cams, num_feat_flatten, feat_dims):
            value (same as key):
            bev_pos (bs, num_query, embed_dims):
            query_pos (None):
            key_pos (None):
            attn_masks (None):
            query_key_padding_mask (None):
            key_padding_mask (None):
            ref_2d (2, num_query, bs, 2):
            ref_3d (bs, num_Z_anchors, num_query, 3):
            bev_h (int):
            bev_w (int):
            reference_points_cam (num_cams, bs, num_query, num_Z_anchors, 2):
            bev_mask (num_cams, bs, num_query, num_Z_anchors):
            mask (None):
            spatial_shapes (num_levels, 2):
            level_start_index (num_levels):
            prev_bev ():
            num_bev_queue (int):
            **kwargs ():

            query List: [(bs, num_query, embed_dims)]:
            key List: [(num_cams, num_feat_flatten, feat_dims)]:
            value List: [(same as key)]:
            bev_pos List: [(bs, num_query, embed_dims)]:
            query_pos (None):
            key_pos (None):
            attn_masks (None):
            query_key_padding_mask (None):
            key_padding_mask (None):
            ref_2d (2, num_query, bs, 2):
            ref_3d (bs, num_Z_anchors, num_query, 3):
            bev_h (int):
            bev_w (int):
            reference_points_cam List [(num_cams, bs, num_query, num_Z_anchors, 2)]:
            bev_mask List [(num_cams, bs, num_query, num_Z_anchors)]:
            mask (None):
            spatial_shapes (num_levels, 2):
            level_start_index (num_levels):
            prev_bev ():
            num_bev_queue (int):
            **kwargs ():
        Returns:
        '''
        if not isinstance(query, (list, tuple)):
            return self.get_single_bev(
                query,
                key,
                value,
                bev_pos,
                ref_2d,
                reference_points_cam,
                bev_mask,
                bev_h,
                bev_w,
                spatial_shapes,
                level_start_index,
                **kwargs
            )

    def get_single_bev(self,
                       query,
                       key,
                       value,
                       bev_pos,
                       ref_2d,
                       reference_points_cam,
                       bev_mask,
                       bev_h,
                       bev_w,
                       spatial_shapes,
                       level_start_index,
                       **kwargs):
        assert not isinstance(query, (list, tuple))
        assert not isinstance(key, (list, tuple))
        assert not isinstance(value, (list, tuple))
        assert not isinstance(bev_pos, (list, tuple))
        assert not isinstance(reference_points_cam, (list, tuple))
        assert not isinstance(bev_mask, (list, tuple))

        query = self.attentions[0](
            query,
            query_pos=bev_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor(
                [[bev_h, bev_w]],
                device=query.device
            ),
            level_start_index=torch.tensor(
                [0],
                device=query.device
            ),
            **kwargs
        )
        query = self.norms[0](query)
        query = self.attentions[1](
            query,
            key,
            value,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        identity = query
        query = self.norms[1](query)
        query = self.ffns[0](
            query,
            identity if self.pre_norm else None
        )
        query = self.norms[2](query)
        return query


@TRANSFORMER_LAYER.register_module()
class MapPriorLayer(MyCustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(
                     type='ReLU',
                     inplace=True),
                 norm_cfg=dict(
                     type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(MapPriorLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)

        self.fp16_enabled = False
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(
        #    ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                bev_mask=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                num_bev_queue=2,
                **kwargs):
        '''

        Args:
            query (bs, num_query, embed_dims):
            key (num_cams, num_feat_flatten, feat_dims):
            value (same as key):
            bev_pos (bs, num_query, embed_dims):
            query_pos (None):
            key_pos (None):
            attn_masks (None):
            query_key_padding_mask (None):
            key_padding_mask (None):
            ref_2d (2, num_query, bs, 2):
            ref_3d (bs, num_Z_anchors, num_query, 3):
            bev_h (int):
            bev_w (int):
            reference_points_cam (num_cams, bs, num_query, num_Z_anchors, 2):
            bev_mask (num_cams, bs, num_query, num_Z_anchors):
            mask (None):
            spatial_shapes (num_levels, 2):
            level_start_index (num_levels):
            prev_bev ():
            num_bev_queue (int):
            **kwargs ():

            query List: [(bs, num_query, embed_dims)]:
            key List: [(num_cams, num_feat_flatten, feat_dims)]:
            value List: [(same as key)]:
            bev_pos List: [(bs, num_query, embed_dims)]:
            query_pos (None):
            key_pos (None):
            attn_masks (None):
            query_key_padding_mask (None):
            key_padding_mask (None):
            ref_2d (2, num_query, bs, 2):
            ref_3d (bs, num_Z_anchors, num_query, 3):
            bev_h (int):
            bev_w (int):
            reference_points_cam List [(num_cams, bs, num_query, num_Z_anchors, 2)]:
            bev_mask List [(num_cams, bs, num_query, num_Z_anchors)]:
            mask (None):
            spatial_shapes (num_levels, 2):
            level_start_index (num_levels):
            prev_bev ():
            num_bev_queue (int):
            **kwargs ():
        Returns:
        '''
        if not isinstance(query, (list, tuple)):
            return self.get_single_bev(
                query,
                key,
                value,
                bev_pos,
                ref_2d,
                reference_points_cam,
                bev_mask,
                bev_h,
                bev_w,
                spatial_shapes,
                level_start_index,
                **kwargs
            )

    def get_single_bev(self,
                       query,
                       key,
                       value,
                       bev_pos,
                       ref_2d,
                       reference_points_cam,
                       bev_mask,
                       bev_h,
                       bev_w,
                       spatial_shapes,
                       level_start_index,
                       **kwargs):
        assert not isinstance(query, (list, tuple))
        assert not isinstance(key, (list, tuple))
        assert not isinstance(value, (list, tuple))
        assert not isinstance(bev_pos, (list, tuple))
        assert not isinstance(reference_points_cam, (list, tuple))
        assert not isinstance(bev_mask, (list, tuple))

        # print('miaomiaomiao' * 80)
        # query[..., -4:] = kwargs['img_metas'][0]['global_prior_onehot']
        query = self.attentions[0](
            query,
            query_pos=bev_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor(
                [[bev_h, bev_w]],
                device=query.device
            ),
            level_start_index=torch.tensor(
                [0],
                device=query.device
            ),
            **kwargs
        )
        query = self.norms[0](query)
        query = self.attentions[1](
            query,
            key,
            value,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        identity = query
        query = self.norms[1](query)

        cur_query = query
        query = self.attentions[2](
            cur_query + bev_pos,
            key=kwargs['img_metas'][0]['global_hist_state'],
            bev2win=True,
            win2bev=True,
            **kwargs
        )

        query = self.ffns[0](
            query,
            identity if self.pre_norm else None
        )
        query = self.norms[2](query)
        return query


@TRANSFORMER_LAYER.register_module()
class MapPriorDeformableLayer(MyCustomBaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(
                     type='ReLU',
                     inplace=True),
                 norm_cfg=dict(
                     type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(MapPriorDeformableLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)

        self.fp16_enabled = False
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(
        #    ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                bev_mask=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                num_bev_queue=2,
                **kwargs):
        '''

        Args:
            query (bs, num_query, embed_dims):
            key (num_cams, num_feat_flatten, feat_dims):
            value (same as key):
            bev_pos (bs, num_query, embed_dims):
            query_pos (None):
            key_pos (None):
            attn_masks (None):
            query_key_padding_mask (None):
            key_padding_mask (None):
            ref_2d (2, num_query, bs, 2):
            ref_3d (bs, num_Z_anchors, num_query, 3):
            bev_h (int):
            bev_w (int):
            reference_points_cam (num_cams, bs, num_query, num_Z_anchors, 2):
            bev_mask (num_cams, bs, num_query, num_Z_anchors):
            mask (None):
            spatial_shapes (num_levels, 2):
            level_start_index (num_levels):
            prev_bev ():
            num_bev_queue (int):
            **kwargs ():

            query List: [(bs, num_query, embed_dims)]:
            key List: [(num_cams, num_feat_flatten, feat_dims)]:
            value List: [(same as key)]:
            bev_pos List: [(bs, num_query, embed_dims)]:
            query_pos (None):
            key_pos (None):
            attn_masks (None):
            query_key_padding_mask (None):
            key_padding_mask (None):
            ref_2d (2, num_query, bs, 2):
            ref_3d (bs, num_Z_anchors, num_query, 3):
            bev_h (int):
            bev_w (int):
            reference_points_cam List [(num_cams, bs, num_query, num_Z_anchors, 2)]:
            bev_mask List [(num_cams, bs, num_query, num_Z_anchors)]:
            mask (None):
            spatial_shapes (num_levels, 2):
            level_start_index (num_levels):
            prev_bev ():
            num_bev_queue (int):
            **kwargs ():
        Returns:
        '''
        if not isinstance(query, (list, tuple)):
            return self.get_single_bev(
                query,
                key,
                value,
                bev_pos,
                ref_2d,
                reference_points_cam,
                bev_mask,
                bev_h,
                bev_w,
                spatial_shapes,
                level_start_index,
                **kwargs
            )

    def get_single_bev(self,
                       query,
                       key,
                       value,
                       bev_pos,
                       ref_2d,
                       reference_points_cam,
                       bev_mask,
                       bev_h,
                       bev_w,
                       spatial_shapes,
                       level_start_index,
                       **kwargs):
        assert not isinstance(query, (list, tuple))
        assert not isinstance(key, (list, tuple))
        assert not isinstance(value, (list, tuple))
        assert not isinstance(bev_pos, (list, tuple))
        assert not isinstance(reference_points_cam, (list, tuple))
        assert not isinstance(bev_mask, (list, tuple))

        query = self.attentions[0](
            query,
            query_pos=bev_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor(
                [[bev_h, bev_w]],
                device=query.device
            ),
            level_start_index=torch.tensor(
                [0],
                device=query.device
            ),
            **kwargs
        )
        query = self.norms[0](query)
        query = self.attentions[1](
            query,
            key,
            value,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )
        query = self.norms[1](query)

        cur_query = query
        query = self.attentions[2](
            # cur_query + bev_pos,
            cur_query,
            key=kwargs['img_metas'][0]['global_hist_state'],
            query_pos=bev_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor(
                [[bev_h, bev_w]],
                device=query.device
            ),
            level_start_index=torch.tensor(
                [0],
                device=query.device
            ),
            **kwargs
        )
        identity = query
        query = self.norms[2](query)

        # cur_query = query
        # query = self.attentions[2](
        #     cur_query + bev_pos,
        #     key=kwargs['img_metas'][0]['global_hist_state'],
        #     bev2win=True,
        #     win2bev=True,
        #     **kwargs
        # )

        query = self.ffns[0](
            query,
            identity if self.pre_norm else None
        )
        query = self.norms[3](query)
        return query

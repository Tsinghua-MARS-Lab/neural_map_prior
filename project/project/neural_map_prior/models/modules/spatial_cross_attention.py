# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import warnings

import math
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION)
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.runner import force_fp32
from mmcv.runner.base_module import BaseModule
from mmcv.utils import ext_loader

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.pc_range = pc_range
        self.dropout = nn.Dropout(dropout)
        self.init_cfg = init_cfg
        self.batch_first = batch_first
        self.deformable_attention = build_attention(deformable_attention)

        self.fp16_enabled = False
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                reference_points_cam=None,
                bev_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                bs=1,
                flag='encoder',
                **kwargs):
        '''
        Args:
            query (bs, num_query, embed_dims):
            key (num_cams, num_feat_flatten, feat_dims):
            value (same as key):
            spatial_shapes (num_levels, 2):
            reference_points_cam (num_cams, bs, num_query, num_Z_anchors, 2):
            bev_mask (num_cams, bs, num_query, num_Z_anchors):
            level_start_index (num_levels):
            bs (int):
            flag ():
            **kwargs ():

        Returns:

        '''

        num_Z_anchors = reference_points_cam.size(3)

        if key is None:
            key = query

        if value is None:
            value = key

        inp_residual = query
        slots = torch.zeros_like(query)

        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [self.num_cams, max_len, self.embed_dims]
        )
        reference_points_rebatch = reference_points_cam.new_zeros(
            [self.num_cams, max_len, num_Z_anchors, 2]
        )

        for i, reference_points_per_img in enumerate(reference_points_cam):
            for j in range(bs):
                index_query_per_img = indexes[i]
                queries_rebatch[j * self.num_cams + i, :len(index_query_per_img)] \
                    = query[j, index_query_per_img]
                reference_points_rebatch[j * self.num_cams + i, :len(index_query_per_img)] \
                    = reference_points_per_img[j, index_query_per_img]

        queries = self.deformable_attention(
            query=queries_rebatch,
            key=key,
            value=value,
            reference_points=reference_points_rebatch,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        for i, index_query_per_img in enumerate(indexes):
            for j in range(bs):
                slots[j, index_query_per_img] += \
                    queries[j * self.num_cams + i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.batch_first = batch_first
        self.norm_cfg = norm_cfg

        self.output_proj = None
        self.fp16_enabled = False

        self.sampling_offsets = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(
            embed_dims, embed_dims
        )

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)

        thetas = torch.arange(
            self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack(
            [thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(
            -1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1,
                                            self.num_levels,
                                            self.num_points,
                                            1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        '''

        Args:
            query (num_cams, query_max_len, embed_dims):
            key (num_cams, num_feat_flatten, feat_dims):
            value (same as key):
            identity (None):
            query_pos (None):
            key_padding_mask (None):
            reference_points (num_cams, query_max_len, num_Z_anchors, 2):
            spatial_shapes (num_levels, 2):
            level_start_index (num_levels):
            **kwargs ():

        Returns:

        '''

        if value is None:
            value = query

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points
        )

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1],
                 spatial_shapes[..., 0]],
                -1
            )  # (num_levels, 2)
            sampling_offsets /= offset_normalizer[None, None, None, :, None, :]

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape

            sampling_offsets = sampling_offsets.view(
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points // num_Z_anchors,
                num_Z_anchors,
                xy
            )

            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_locations = reference_points + sampling_offsets
            # sampling_locations = sampling_offsets

            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points,
                xy
            )

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  value: bs, num_value, num_heads, feat_dims // num_head
        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32

            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value,
                spatial_shapes,
                sampling_locations,
                attention_weights)

        return output

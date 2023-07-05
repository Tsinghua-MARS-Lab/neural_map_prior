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
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.runner.base_module import BaseModule
from mmcv.utils import ext_loader

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class PriorCrossAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 use_current_prior_value=False,
                 use_prior_value=False,
                 use_current_value=False,
                 use_current_query=False,
                 use_current_prior_query=False,
                 ):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.use_current_prior_value = use_current_prior_value
        self.use_prior_value = use_prior_value
        self.use_current_value = use_current_value
        self.use_current_query = use_current_query
        self.use_current_prior_query = use_current_prior_query

        if self.use_current_query:
            self.num_bev_queue_query = 1
        elif self.use_current_prior_query:
            self.num_bev_queue_query = 2

        if self.use_prior_value or self.use_current_value:
            self.num_bev_queue_value = 1
        elif self.use_current_prior_value:
            self.num_bev_queue_value = 2

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

        self.sampling_offsets = nn.Linear(
            embed_dims * self.num_bev_queue_query,
            self.num_bev_queue_value * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims * self.num_bev_queue_query,
            self.num_bev_queue_value * num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels * self.num_bev_queue_value, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                query_pos=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        '''

        Args:
            query (bs, num_query, embed_dims):
            query_pos (bs, num_query, embed_dims):
            reference_points (2, num_query, bs, 2):
            spatial_shapes (bev_h, bev_w):
            level_start_index (0):
            flag ():
            **kwargs ():

        Returns:

        '''

        identity = query

        if self.use_current_prior_value:
            key = kwargs['key']
            value = torch.cat([query, key], 0)
        elif self.use_prior_value:
            value = kwargs['key']
        elif self.use_current_value:
            value = query

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        if self.use_current_query:
            query = query
        elif self.use_current_prior_query:
            key = kwargs['key']
            query = torch.cat([query, key], -1)

        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        value = self.value_proj(value)
        value = value.reshape(self.num_bev_queue_value * bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue_value,
            self.num_levels,
            self.num_points,
            2)
        attention_weights = self.attention_weights(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue_value,
            self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query,
            self.num_heads,
            self.num_bev_queue_value,
            self.num_levels,
            self.num_points
        )

        attention_weights = attention_weights.permute(3, 0, 1, 2, 4, 5) \
            .reshape(bs * self.num_bev_queue_value, num_query, self.num_heads, self.num_levels,
                     self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(3, 0, 1, 2, 4, 5, 6) \
            .reshape(bs * self.num_bev_queue_value, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2 and self.num_bev_queue_value == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1],
                 spatial_shapes[..., 0]],
                -1
            )
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets \
                                 / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points \
                                 * reference_points[:, :, None, :, None, 2:] \
                                 * 0.5
        elif reference_points.shape[-1] == 2 and self.num_bev_queue_value == 1:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1],
                 spatial_shapes[..., 0]],
                -1
            )
            sampling_locations = reference_points[:1, :, None, :, None, :] \
                                 + sampling_offsets \
                                 / offset_normalizer[None, None, None, :, None, :]

        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
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

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs)
        if self.num_bev_queue_value == 2:
            output = (output[..., :bs] + output[..., bs:]) / self.num_bev_queue

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        return self.dropout(output) + identity

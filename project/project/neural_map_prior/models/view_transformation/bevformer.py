import torch
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torch import nn
from torch.nn.init import normal_

from mmdet3d.models.builder import NECKS
from project.neural_map_prior.models.modules.decoder import CustomMSDeformableAttention
from project.neural_map_prior.models.modules.spatial_cross_attention import MSDeformableAttention3D
from project.neural_map_prior.models.modules.temporal_self_attention import TemporalSelfAttention


@NECKS.register_module()
class BEVFormer(nn.Module):
    def __init__(self, embed_dims, num_feature_levels, num_cams, use_cams_embeds,
                 map_bev_attrs, positional_encoding, encoder):
        super(BEVFormer, self).__init__()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_cams_embeds = use_cams_embeds

        self.bev_h = map_bev_attrs['bev_h_']
        self.bev_w = map_bev_attrs['bev_w_']
        self.real_h = map_bev_attrs['real_h_']
        self.real_w = map_bev_attrs['real_w_']

        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.encoder = build_transformer_layer_sequence(encoder)

        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

    def _init_weight(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or \
                    isinstance(m, TemporalSelfAttention) or \
                    isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    def get_bev_query_and_pos(self, imgs):
        """
        Args:
            imgs ():
        Returns:
            torch.Size([1, 20000, 256])
            torch.Size([1, 20000, 256])
        """
        dtype, batch_size = imgs.dtype, imgs.shape[0]
        bev_mask = imgs.new_zeros((batch_size, self.bev_h, self.bev_w))
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1)

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        return bev_queries, bev_pos

    def flatten_feats(self, img_feats):
        """
        Args:
            img_feats ():
        Returns:
        torch.Size([6, 30825, 256])
        torch.float32
        torch.Size([4, 2])
        torch.int64
        torch.Size([4])
        torch.int64
        """
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            bs, num_cams, channel, height, width = feat.size()
            spatial_shape = (height, width)
            # B, num_cams, C, H * W
            feat = feat.flatten(3)
            # num_cams, B, H * W, C
            feat = feat.permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=img_feats[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # num_cams, B, H * W * lvl, C
        feat_flatten = torch.cat(feat_flatten, 2)
        # num_cams, H * W * lvl, B, embed_dims
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        # num_cams, H * W * lvl, embed_dims
        feat_flatten = feat_flatten.squeeze(2)
        return feat_flatten, spatial_shapes, level_start_index

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
            bev_feats(tensor): batch_size, bev_h * bev_w, embed_dims

        """
        feats_flatten, spatial_shapes, level_start_index = self.flatten_feats(imgs_feats)
        bev_queries, bev_pos = self.get_bev_query_and_pos(imgs)
        bev_feats = self.encoder(
            bev_queries,
            feats_flatten,
            feats_flatten,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=imgs.new_tensor([0.0, 0.0]),
            **kwargs
        )
        return bev_feats

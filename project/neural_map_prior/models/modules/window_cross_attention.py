from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import ATTENTION
from scipy.spatial.transform import Rotation as R


# __all__: List[str] = ["SwinTransformerStage", "SwinTransformerBlock", "DeformableSwinTransformerBlock"]
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


class FeedForward(nn.Sequential):
    """
    Feed forward module used in the transformer encoder.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float = 0.) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param hidden_features: (int) Number of hidden features
        :param out_features: (int) Number of output features
        :param dropout: (float) Dropout factor
        """
        # Call super constructor and init modules
        super().__init__(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Dropout(p=dropout)
        )


def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1)


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2)


def unfold(input: torch.Tensor,
           window_size: int) -> torch.Tensor:
    """
    Unfolds (non-overlapping) a given feature map by the given window size (stride = window size)
    :param input: (torch.Tensor) Input feature map of the shape [batch size, channels, height, width]
    :param window_size: (int) Window size to be applied
    :return: (torch.Tensor) Unfolded tensor of the shape [batch size * windows, channels, window size, window size]
    """
    # Get original shape
    _, channels, height, width = input.shape  # type: int, int, int, int
    # Unfold input
    output: torch.Tensor = input.unfold(dimension=3, size=window_size, step=window_size) \
        .unfold(dimension=2, size=window_size, step=window_size)
    # Reshape to [batch size * windows, channels, window size, window size]
    output: torch.Tensor = output.permute(0, 2, 3, 1, 5, 4).reshape(-1, channels, window_size, window_size)
    return output


def fold(input: torch.Tensor,
         window_size: int,
         height: int,
         width: int) -> torch.Tensor:
    """
    Fold a tensor of windows again to a 4D feature map
    :param input: (torch.Tensor) Input tensor of windows [batch size * windows, channels, window size, window size]
    :param window_size: (int) Window size to be reversed
    :param height: (int) Height of the feature map
    :param width: (int) Width of the feature map
    :return: (torch.Tensor) Folded output tensor of the shape [batch size, channels, height, width]
    """
    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    batch_size: int = int(input.shape[0] // (height * width // window_size // window_size))
    # Reshape input to
    output: torch.Tensor = input.view(batch_size, height // window_size, width // window_size, channels,
                                      window_size, window_size)
    output: torch.Tensor = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
    return output


class WindowMultiHeadAttention(nn.Module):
    """
    This class implements window-based Multi-Head-Attention.
    """

    def __init__(self,
                 in_features: int,
                 window_size: int,
                 number_of_heads: int,
                 dropout_attention: float = 0.,
                 dropout_projection: float = 0.,
                 meta_network_hidden_features: int = 256,
                 sequential_self_attention: bool = False) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param window_size: (int) Window size
        :param number_of_heads: (int) Number of attention heads
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_projection: (float) Dropout rate after projection
        :param meta_network_hidden_features: (int) Number of hidden features in the two layer MLP meta network
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(WindowMultiHeadAttention, self).__init__()
        # Check parameter
        assert (in_features % number_of_heads) == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (number_of_heads)."
        # Save parameters
        self.in_features: int = in_features
        self.window_size: int = window_size
        self.number_of_heads: int = number_of_heads
        self.sequential_self_attention: bool = sequential_self_attention
        # Init query, key and value mapping as a single layer
        self.mapping_q: nn.Module = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        self.mapping_kv: nn.Module = nn.Linear(in_features=in_features, out_features=in_features * 2, bias=True)
        # Init attention dropout
        self.attention_dropout: nn.Module = nn.Dropout(dropout_attention)
        # Init projection mapping
        self.projection: nn.Module = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        # Init projection dropout
        self.projection_dropout: nn.Module = nn.Dropout(dropout_projection)
        # Init meta network for positional encodings
        self.meta_network: nn.Module = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_network_hidden_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_network_hidden_features, out_features=number_of_heads, bias=True))
        # Init tau
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(1, number_of_heads, 1, 1)))
        # Init pair-wise relative positions (log-spaced)
        self.__make_pair_wise_relative_positions()

    def __make_pair_wise_relative_positions(self) -> None:
        """
        Method initializes the pair-wise relative positions to compute the positional biases
        """
        indexes: torch.Tensor = torch.arange(self.window_size, device=self.tau.device)
        coordinates: torch.Tensor = torch.stack(torch.meshgrid([indexes, indexes]), dim=0)
        coordinates: torch.Tensor = torch.flatten(coordinates, start_dim=1)
        relative_coordinates: torch.Tensor = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates: torch.Tensor = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log: torch.Tensor = torch.sign(relative_coordinates) \
                                                 * torch.log(1. + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log)

    def update_resolution(self,
                          new_window_size: int,
                          **kwargs: Any) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param kwargs: (Any) Unused
        """
        # Set new window size
        self.window_size: int = new_window_size
        # Make new pair-wise relative positions
        self.__make_pair_wise_relative_positions()

    def __get_relative_positional_encodings(self) -> torch.Tensor:
        """
        Method computes the relative positional encodings
        :return: (torch.Tensor) Relative positional encodings [1, number of heads, window size ** 2, window size ** 2]
        """
        relative_position_bias: torch.Tensor = self.meta_network(self.relative_coordinates_log)
        relative_position_bias: torch.Tensor = relative_position_bias.permute(1, 0)
        relative_position_bias: torch.Tensor = relative_position_bias.reshape(self.number_of_heads,
                                                                              self.window_size * self.window_size,
                                                                              self.window_size * self.window_size)
        return relative_position_bias.unsqueeze(0)

    def __self_attention(self,
                         query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         batch_size_windows: int,
                         tokens: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs standard (non-sequential) scaled cosine self-attention
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        # Compute attention map with scaled cosine attention
        attention_map: torch.Tensor = torch.einsum("bhqd, bhkd -> bhqk", query, key) \
                                      / torch.maximum(torch.norm(query, dim=-1, keepdim=True)
                                                      * torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1),
                                                      torch.tensor(1e-06, device=query.device, dtype=query.dtype))
        attention_map: torch.Tensor = attention_map / self.tau.clamp(min=0.01)
        # Apply relative positional encodings
        attention_map: torch.Tensor = attention_map + self.__get_relative_positional_encodings()
        # Apply mask if utilized
        if mask is not None:
            # number_of_windows: int = mask.shape[0]
            # attention_map: torch.Tensor = attention_map.view(batch_size_windows // number_of_windows, number_of_windows,
            #                                                  self.number_of_heads, tokens, tokens)
            # attention_map: torch.Tensor = attention_map + mask.unsqueeze(1).unsqueeze(0)
            # attention_map: torch.Tensor = attention_map.view(-1, self.number_of_heads, tokens, tokens)
            attention_map = attention_map + mask
        attention_map: torch.Tensor = attention_map.softmax(dim=-1)
        # Perform attention dropout
        attention_map: torch.Tensor = self.attention_dropout(attention_map)
        # Apply attention map and reshape
        output: torch.Tensor = torch.einsum("bhal, bhlv -> bhav", attention_map, value)
        output: torch.Tensor = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        return output

    def forward(self,
                input: torch.Tensor,
                key: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size * windows, channels, height, width]
        :param key: (torch.Tensor) Input tensor of the shape [batch size * windows, channels, height, width]
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output tensor of the shape [batch size * windows, channels, height, width]
        """
        # Save original shape
        batch_size_windows, channels, height, width = input.shape  # type: int, int, int, int
        tokens: int = height * width
        # Reshape input to [batch size * windows, tokens (height * width), channels]
        input: torch.Tensor = input.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
        key: torch.Tensor = key.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
        if mask is not None:
            mask = mask.reshape(batch_size_windows, 1, tokens).unsqueeze(3)
        # Perform query, key, and value mapping
        query: torch.Tensor = self.mapping_q(input)
        key_value: torch.Tensor = self.mapping_kv(key)
        query: torch.Tensor = query.view(batch_size_windows, tokens, self.number_of_heads,
                                         channels // self.number_of_heads).permute(0, 2, 1, 3)
        key_value: torch.Tensor = key_value.view(batch_size_windows, tokens, 2, self.number_of_heads,
                                                 channels // self.number_of_heads).permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        output: torch.Tensor = self.__self_attention(
            query=query, key=key, value=value,
            batch_size_windows=batch_size_windows,
            tokens=tokens, mask=mask)
        # Perform linear mapping and dropout
        output: torch.Tensor = self.projection_dropout(self.projection(output))
        # Reshape output to original shape [batch size * windows, channels, height, width]
        output: torch.Tensor = output.permute(0, 2, 1).view(batch_size_windows, channels, height, width)
        return output


@ATTENTION.register_module()
class WindowCrossAttention(nn.Module):
    def __init__(self,
                 num_bev_win_h=20,
                 num_bev_win_w=40,
                 bev_h=100,
                 bev_w=200,
                 embed_dims=256,
                 num_heads=8,
                 num_bev_queue=2,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_size=1):
        super().__init__()
        if bev_h % num_bev_win_h != 0:
            raise ValueError(f'bev_h must be divisible by num_bev_win_h, '
                             f'but got {bev_h} and {num_bev_win_h}')
        if bev_w % num_bev_win_w != 0:
            raise ValueError(f'bev_w must be divisible by num_bev_win_w, '
                             f'but got {bev_w} and {num_bev_win_w}')

        self.num_bev_win_h = num_bev_win_h
        self.num_bev_win_w = num_bev_win_w
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_win_size_h = self.bev_h // self.num_bev_win_h
        self.bev_win_size_w = self.bev_w // self.num_bev_win_w
        # assert self.bev_win_size_h == self.bev_win_size_w
        self.window_size = self.bev_win_size_h
        self.batch_size = batch_size
        self.embed_dims = embed_dims
        self.window_attention: WindowMultiHeadAttention = WindowMultiHeadAttention(
            in_features=256,
            window_size=self.window_size,
            number_of_heads=8,
            dropout_attention=0.0,
            dropout_projection=0.0,
            sequential_self_attention=False)
        self.normalization_1: nn.Module = nn.LayerNorm(normalized_shape=embed_dims)

    def forward(self,
                query,
                key,
                bev2win=False,
                win2bev=False,
                hist_index=None,
                **kwargs):
        '''
        Args:
            query (batch size, self.bev_h * self.bev_w, in channels):
            key (batch size self.bev_h * self.bev_w, in channels):
            bev2win (bool):
            win2bev (bool):
            **kwargs ():

        Returns:
           (batch size, self.bev_h * self.bev_w, in channels)
        '''

        _, _, C = query.size()
        # [batch size, in channels, height, width]
        query = query.reshape(self.batch_size, self.bev_h, self.bev_w, C)

        _, _, C = key.size()
        # [batch size, in channels, height, width]
        key = key.reshape(self.batch_size, self.bev_h, self.bev_w, C)

        attention_mask = torch.zeros_like(query).detach()
        attention_mask = attention_mask.masked_fill(key == 0, float(-100.0))[..., 0:1]

        query = query.permute(0, 3, 1, 2)
        key = key.permute(0, 3, 1, 2)

        batch_size, channels, height, width = query.shape  # type: int, int, int, int
        # Make patches
        # torch.Size([200, 256, 10, 10])
        query_patches: torch.Tensor = unfold(input=query, window_size=self.window_size)
        # torch.Size([200, 256, 10, 10])
        key_patches: torch.Tensor = unfold(input=key, window_size=self.window_size)

        if attention_mask is not None:
            attention_mask = attention_mask.permute(0, 3, 1, 2)
            attention_mask: torch.Tensor = unfold(input=attention_mask, window_size=self.window_size)
        # Perform window attention
        output_attention: torch.Tensor = self.window_attention(
            query_patches, key_patches, mask=attention_mask)
        # Merge patches
        output_merge: torch.Tensor = fold(
            input=output_attention, window_size=self.window_size, height=height, width=width)

        # Perform normalization
        output_normalize: torch.Tensor = self.normalization_1(output_merge.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # Skip connection
        # output_skip: torch.Tensor = output_normalize + query
        output_skip: torch.Tensor = output_normalize + query
        output_skip = output_skip.flatten(2, 3).permute(0, 2, 1)
        return output_skip

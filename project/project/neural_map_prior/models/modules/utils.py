import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from scipy.spatial.transform import Rotation as R


def gen_matrix(ego2global_rotation, ego2global_translation):
    rotation_xyz = np.roll(ego2global_rotation, shift=-1)
    trans = np.eye(4)
    trans[:3, 3] = ego2global_translation
    trans[:3, :3] = R.from_quat(rotation_xyz).as_matrix()
    return trans


def gen_ego2ego_matrix(ori_ego2global_rotation, ref_ego2global_rotation,
                       ori_ego2global_translation, ref_ego2global_translation):
    ori_trans = gen_matrix(ori_ego2global_rotation, ori_ego2global_translation)
    ref_trans = gen_matrix(ref_ego2global_rotation, ref_ego2global_translation)
    return np.linalg.inv(ori_trans) @ ref_trans


def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
    """Get the reference points used in SCA and TSA.
    """

    # reference points in 3D space, used in spatial cross-attention (SCA)
    if dim == '3d':
        zs = torch.linspace(
            0.5, Z - 0.5, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = torch.flatten(ref_3d, start_dim=1, end_dim=2)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)

        return ref_3d.to(dtype).to(device)

    # reference points on 2D bev plane, used in temporal self-attention (TSA).
    elif dim == '2d':
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d


# This function must use fp32!!!
# @force_fp32(apply_to=('reference_points', 'img_metas'))
def point_sampling(reference_points, pc_range, img_metas, hist_pointer=None):
    '''

    Args:
        reference_points (bs, num_Z_anchors, num_query, 3):
        pc_range (list: [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]):
        img_metas ():

    Returns:
        (bs, num_Z_anchors, num_query, 3)
        (bs, num_cams, 4, 4) -->
        (num_cams, bs, num_query, num_Z_anchors, 2)
    '''

    batch_ego2img = []
    for img_meta in img_metas:
        if hist_pointer is None:
            ego2img = np.array(img_meta['ego2img'])
        elif hist_pointer >= 0:
            num_cams = 6
            ego2img = np.array(img_meta['ego2img_history'])[hist_pointer * num_cams:(hist_pointer + 1) * num_cams]
            print(f'point_sampling_{hist_pointer}')
            assert ego2img.shape[0] == 6
            assert ego2img.shape[1] == 4
            assert ego2img.shape[2] == 4
        # if 'num_history_frame' in img_meta:
        #     num_hist = 5
        #     ego2img = ego2img[None].repeat(num_hist + 1, axis=0).reshape(-1, 4, 4)
        #     if img_meta['num_history_frame'] > 0:
        #         his_ego2img = np.array(img_meta['ego2img_history'])
        #
        #         cur_ego2global_translation = img_meta['ego2global_translation']
        #         cur_ego2global_rotation = img_meta['ego2global_rotation']
        #         his_ego2global_translation = img_meta['ego2global_translation_history']
        #         his_ego2global_rotation = img_meta['ego2global_rotation_history']
        #
        #         ego2ego = []
        #         for i in range(len(his_ego2global_translation)):
        #             ego2ego.append(
        #                 gen_ego2ego_matrix(
        #                     his_ego2global_rotation[i],
        #                     cur_ego2global_rotation,
        #                     his_ego2global_translation[i],
        #                     cur_ego2global_translation))
        #
        #         ego2ego = np.array(ego2ego)[:, None]
        #         num_cams = len(his_ego2img) // len(ego2ego)
        #         ego2ego = ego2ego.repeat(num_cams, axis=1).reshape(-1, 4, 4)
        #
        #         assert his_ego2img.shape == ego2ego.shape
        #         his_ego2ego2img = np.einsum('ijk,ikl->ijl', his_ego2img, ego2ego)
        #         ego2img[num_cams:(img_meta['num_history_frame'] + 1) * num_cams] = his_ego2ego2img
        batch_ego2img.append(ego2img)

    ego2img = np.array(batch_ego2img)
    ego2img = reference_points.new_tensor(ego2img)  # (B, N, 4, 4)
    lidar2img = ego2img

    reference_points = reference_points.clone()

    reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                 (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                 (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                 (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                        reference_points.to(torch.float32)).squeeze(-1)
    eps = 1e-5

    bev_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))
    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        bev_mask = torch.nan_to_num(bev_mask)
    else:
        bev_mask = bev_mask.new_tensor(
            np.nan_to_num(bev_mask.cpu().numpy()))

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    return reference_points_cam, bev_mask


# This function must use fp32!!!
# @force_fp32(apply_to=('reference_points', 'img_metas'))
def new_point_sampling(self, reference_points, pc_range, img_metas):
    '''

    Args:
        reference_points (bs, num_Z_anchors, num_query, 3):
        pc_range (list: [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]):
        img_metas ():

    Returns:
        (bs, num_Z_anchors, num_query, 3)
        (bs, num_cams, 4, 4) -->
        (num_cams, bs, num_query, num_Z_anchors, 2)
    '''
    # (bs, num_cams, 4, 4)
    ego2img = [img_meta['ego2img'] for img_meta in img_metas]
    ego2img = np.asarray(ego2img)
    lidar2img = reference_points.new_tensor(ego2img)
    assert lidar2img.dtype == torch.float32

    reference_points = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)

    reference_points_cam = torch.einsum('bcij,baqj->bcaqi', lidar2img, reference_points)
    reference_points_cam = reference_points_cam.permute(1, 0, 3, 2, 4)

    eps = 1e-5
    bev_mask = (reference_points_cam[..., 2:3] > eps)

    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3],
        torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))

    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        bev_mask = torch.nan_to_num(bev_mask)
    else:
        bev_mask = bev_mask.new_tensor(
            np.nan_to_num(bev_mask.cpu().numpy()))

    bev_mask = bev_mask.squeeze(-1)
    return reference_points_cam, bev_mask

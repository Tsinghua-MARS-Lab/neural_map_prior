import numpy as np
import torch
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .base_nusc_dataset import NuscData
from .nuscene import MyNuScenesMap, extract_contour
from .utils import gen_dx_bx, label_onehot_encoding

MAP = ['boston-seaport', 'singapore-hollandvillage',
       'singapore-onenorth', 'singapore-queenstown']


class RaseterizedData(NuscData):

    def __init__(self,
                 raw_dataset_cfg,
                 data_aug_conf,
                 grid_conf):

        self.nusc = NuScenes(version='{}'.format(raw_dataset_cfg['version']),
                             dataroot=raw_dataset_cfg['data_root'],
                             verbose=False)
        nusc_maps = {}
        for map_name in MAP:
            nusc_maps[map_name] = MyNuScenesMap(
                dataroot=raw_dataset_cfg['data_root'], map_name=map_name)
        self.nusc_maps = nusc_maps

        self.thickness = data_aug_conf['line_width']
        self.angle_class = data_aug_conf['angle_class']
        self.direction_pred = data_aug_conf['direction_pred']

        dx, bx, nx = gen_dx_bx(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        patch_h = grid_conf['ybound'][1] - grid_conf['ybound'][0]
        patch_w = grid_conf['xbound'][1] - grid_conf['xbound'][0]
        canvas_h = int(patch_h / grid_conf['ybound'][2])
        canvas_w = int(patch_w / grid_conf['xbound'][2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)

    def get_lineimg(self, pose_ego, location, forward_backward=False):
        line_seg_layers = ['road_divider', 'lane_divider', 'ped_crossing_line']
        lane_seg_layers = ['road_segment', 'lane']

        line_mask, line_inst = gen_topdown_mask(self.nusc_maps, pose_ego, location, self.patch_size, self.canvas_size,
                                                seg_layers=line_seg_layers, thickness=self.thickness)
        lane_mask, lane_inst = gen_topdown_mask(self.nusc_maps, pose_ego, location, self.patch_size, self.canvas_size,
                                                seg_layers=lane_seg_layers, thickness=self.thickness)

        cum_inst = np.cumsum(line_inst)
        for i in range(1, line_mask.shape[0]):
            line_mask[i][line_mask[i] != 0] += cum_inst[i - 1]

        contour_mask, contour_inst = extract_contour(np.any(lane_mask, 0).astype('uint8'), self.canvas_size,
                                                     thickness=self.thickness)
        contour_thick_mask, _ = extract_contour(np.any(lane_mask, 0).astype('uint8'), self.canvas_size,
                                                thickness=self.thickness + 4)
        contour_mask[contour_mask != 0] += cum_inst[-1]

        # inst_mask = np.sum(line_mask, 0).astype('int32')
        # inst_mask[contour_thick_mask != 0] = 0
        # inst_mask[contour_mask != 0] = contour_mask[contour_mask != 0]

        inst_mask = np.zeros(
            (4, self.canvas_size[0], self.canvas_size[1]), dtype='uint8')
        inst_mask[3] = contour_mask
        inst_mask[2] = line_mask[2]
        inst_mask[2][contour_thick_mask != 0] = 0
        inst_mask[1] = np.sum(line_mask[:2], axis=0)
        inst_mask[1][(inst_mask[2] != 0) | (contour_thick_mask != 0)] = 0

        seg_mask = np.zeros(
            (4, self.canvas_size[0], self.canvas_size[1]), dtype='uint8')
        seg_mask[3] = contour_mask != 0
        seg_mask[2] = (line_mask[2] != 0) & (contour_thick_mask == 0)
        seg_mask[1] = np.any(line_mask[:2], axis=0) & (
                seg_mask[2] == 0) & (contour_thick_mask == 0)
        seg_mask[0] = 1 - np.any(seg_mask, axis=0)

        if not self.direction_pred:
            return torch.Tensor(seg_mask), torch.Tensor(inst_mask), 0

        line_forward_mask, _ = gen_topdown_mask(self.nusc_maps, pose_ego, location, self.patch_size, self.canvas_size,
                                                seg_layers=line_seg_layers, thickness=self.thickness + 4,
                                                type='forward', angle_class=self.angle_class)
        contour_forward_mask, _ = extract_contour(np.any(lane_mask, 0).astype(
            'uint8'), self.canvas_size, thickness=self.thickness + 4, type='forward', angle_class=self.angle_class)

        forward_mask = np.zeros((4, self.canvas_size[0], self.canvas_size[1]))
        forward_mask[3] = contour_forward_mask
        forward_mask[2] = line_forward_mask[2]
        forward_mask[2][contour_thick_mask != 0] = 0
        forward_mask[1] = np.sum(line_forward_mask[:2], axis=0)
        forward_mask[1][(forward_mask[2] != 0) | (contour_thick_mask != 0)] = 0
        forward_mask = forward_mask.sum(0)

        line_backward_mask, _ = gen_topdown_mask(self.nusc_maps, pose_ego, location, self.patch_size, self.canvas_size,
                                                 seg_layers=line_seg_layers, thickness=self.thickness + 4,
                                                 type='backward', angle_class=self.angle_class)
        contour_backward_mask, _ = extract_contour(np.any(lane_mask, 0).astype(
            'uint8'), self.canvas_size, thickness=self.thickness + 4, type='backward', angle_class=self.angle_class)
        backward_mask = np.zeros((4, self.canvas_size[0], self.canvas_size[1]))
        backward_mask[3] = contour_backward_mask
        backward_mask[2] = line_backward_mask[2]
        backward_mask[2][contour_thick_mask != 0] = 0
        backward_mask[1] = np.sum(line_backward_mask[:2], axis=0)
        backward_mask[1][(backward_mask[2] != 0) |
                         (contour_thick_mask != 0)] = 0
        backward_mask = backward_mask.sum(0)

        if forward_backward:
            return torch.Tensor(seg_mask), torch.Tensor(inst_mask), torch.Tensor(forward_mask), torch.Tensor(
                backward_mask)
        forward_mask = label_onehot_encoding(
            torch.tensor(forward_mask), self.angle_class + 1)
        backward_mask = label_onehot_encoding(
            torch.tensor(backward_mask), self.angle_class + 1)
        direction_mask = forward_mask
        direction_mask[backward_mask != 0] = 1.
        direction_mask = direction_mask / direction_mask.sum(0, keepdim=True)
        return torch.Tensor(seg_mask), torch.Tensor(inst_mask), torch.Tensor(direction_mask)

    def __call__(self, pose_ego, location):
        x = self.get_lineimg(pose_ego, location)
        return x


def gen_topdown_mask(nusc_maps, pose_record, location, patch_size, canvas_size, seg_layers, thickness=5, type='index',
                     angle_class=36):
    map_pose = pose_record['translation'][:2]
    rotation = Quaternion(pose_record['rotation'])

    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    topdown_seg_mask, num_inst = nusc_maps[location].get_map_mask(
        patch_box, patch_angle, seg_layers, canvas_size, thickness=thickness, type=type, angle_class=angle_class)
    # topdown_seg_mask = np.flip(topdown_seg_mask, 1)  # left-right correction
    return topdown_seg_mask, num_inst

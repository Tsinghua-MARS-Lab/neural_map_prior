import os
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from .nuscene import gen_topdown_mask, extract_contour
from .utils import get_lidar_data, img_transform, normalize_img, gen_dx_bx, \
    label_onehot_encoding

MAP = ['boston-seaport', 'singapore-hollandvillage',
       'singapore-onenorth', 'singapore-queenstown']


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, nusc_maps, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        self.preprocess = data_aug_conf['preprocess']
        self.thickness = data_aug_conf['line_width']
        self.angle_class = data_aug_conf['angle_class']
        self.direction_pred = data_aug_conf['direction_pred']

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()
        patch_h = grid_conf['ybound'][1] - grid_conf['ybound'][0]
        patch_w = grid_conf['xbound'][1] - grid_conf['xbound'][0]
        canvas_h = int(patch_h / grid_conf['ybound'][2])
        canvas_w = int(patch_w / grid_conf['xbound'][2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)

        self.fix_nuscenes_formatting()

        # print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        resize = (fW / W, fH / H)
        resize_dims = (fW, fH)
        crop = None
        if self.is_train:
            # resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            # resize_dims = (int(W*resize), int(H*resize))
            # newW, newH = resize_dims
            # crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            # crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            # crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            # if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
            #     flip = True
            # rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
            rotate = 0
        else:
            # resize = max(fH/H, fW/W)
            # resize_dims = (int(W*resize), int(H*resize))
            # newW, newH = resize_dims
            # crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            # crop_w = int(max(0, newW - fW) / 2)
            # crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        color_jitter = torchvision.transforms.ColorJitter.get_params(
            brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=[-0.2, 0.2]
        )

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        sample_data = rec['data']
        sample_data_record = self.nusc.get('sample_data', sample_data['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        translation = pose_record['translation']
        translation = torch.tensor(translation)

        pos_rotation = Quaternion(pose_record['rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        yaw_pitch_roll = torch.tensor(yaw_pitch_roll)

        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            points, depth, _ = self.nusc.explorer.map_pointcloud_to_image(rec['data']['LIDAR_TOP'], rec['data'][cam])
            points, intensity, _ = self.nusc.explorer.map_pointcloud_to_image(rec['data']['LIDAR_TOP'],
                                                                              rec['data'][cam], render_intensity=True)

            # if self.is_train:
            #     img = color_jitter(img)
            img = normalize_img(img)
            # if self.is_train:
            #     img = random_erasing(img)
            imgs.append(img)
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans), translation, yaw_pitch_roll)

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                             nsweeps=nsweeps, min_distance=2.2)
        # return torch.Tensor(pts)# [:3]  # x,y,z
        return pts

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def get_lineimg(self, rec, forward_backward=False):
        line_seg_layers = ['road_divider', 'lane_divider', 'ped_crossing_line']
        lane_seg_layers = ['road_segment', 'lane']

        line_mask, line_inst = gen_topdown_mask(self.nusc, self.nusc_maps, rec, self.patch_size, self.canvas_size,
                                                seg_layers=line_seg_layers, thickness=self.thickness)
        lane_mask, lane_inst = gen_topdown_mask(self.nusc, self.nusc_maps, rec, self.patch_size, self.canvas_size,
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

        inst_mask = np.zeros((4, self.canvas_size[0], self.canvas_size[1]), dtype='uint8')
        inst_mask[3] = contour_mask
        inst_mask[2] = line_mask[2]
        inst_mask[2][contour_thick_mask != 0] = 0
        inst_mask[1] = np.sum(line_mask[:2], axis=0)
        inst_mask[1][(inst_mask[2] != 0) | (contour_thick_mask != 0)] = 0

        seg_mask = np.zeros((4, self.canvas_size[0], self.canvas_size[1]), dtype='uint8')
        seg_mask[3] = contour_mask != 0
        seg_mask[2] = (line_mask[2] != 0) & (contour_thick_mask == 0)
        seg_mask[1] = np.any(line_mask[:2], axis=0) & (seg_mask[2] == 0) & (contour_thick_mask == 0)
        seg_mask[0] = 1 - np.any(seg_mask, axis=0)

        if not self.direction_pred:
            return torch.Tensor(seg_mask), torch.Tensor(inst_mask), 0

        line_forward_mask, _ = gen_topdown_mask(self.nusc, self.nusc_maps, rec, self.patch_size, self.canvas_size,
                                                seg_layers=line_seg_layers, thickness=self.thickness + 4,
                                                type='forward', angle_class=self.angle_class)
        contour_forward_mask, _ = extract_contour(np.any(lane_mask, 0).astype('uint8'), self.canvas_size,
                                                  thickness=self.thickness + 4, type='forward',
                                                  angle_class=self.angle_class)

        forward_mask = np.zeros((4, self.canvas_size[0], self.canvas_size[1]))
        forward_mask[3] = contour_forward_mask
        forward_mask[2] = line_forward_mask[2]
        forward_mask[2][contour_thick_mask != 0] = 0
        forward_mask[1] = np.sum(line_forward_mask[:2], axis=0)
        forward_mask[1][(forward_mask[2] != 0) | (contour_thick_mask != 0)] = 0
        forward_mask = forward_mask.sum(0)

        line_backward_mask, _ = gen_topdown_mask(self.nusc, self.nusc_maps, rec, self.patch_size, self.canvas_size,
                                                 seg_layers=line_seg_layers, thickness=self.thickness + 4,
                                                 type='backward', angle_class=self.angle_class)
        contour_backward_mask, _ = extract_contour(np.any(lane_mask, 0).astype('uint8'), self.canvas_size,
                                                   thickness=self.thickness + 4, type='backward',
                                                   angle_class=self.angle_class)
        backward_mask = np.zeros((4, self.canvas_size[0], self.canvas_size[1]))
        backward_mask[3] = contour_backward_mask
        backward_mask[2] = line_backward_mask[2]
        backward_mask[2][contour_thick_mask != 0] = 0
        backward_mask[1] = np.sum(line_backward_mask[:2], axis=0)
        backward_mask[1][(backward_mask[2] != 0) | (contour_thick_mask != 0)] = 0
        backward_mask = backward_mask.sum(0)

        if forward_backward:
            return torch.Tensor(seg_mask), torch.Tensor(inst_mask), torch.Tensor(forward_mask), torch.Tensor(
                backward_mask)
        forward_mask = label_onehot_encoding(torch.tensor(forward_mask), self.angle_class + 1)
        backward_mask = label_onehot_encoding(torch.tensor(backward_mask), self.angle_class + 1)
        direction_mask = forward_mask
        direction_mask[backward_mask != 0] = 1.
        direction_mask = direction_mask / direction_mask.sum(0, keepdim=True)
        return torch.Tensor(seg_mask), torch.Tensor(inst_mask), torch.Tensor(direction_mask)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)

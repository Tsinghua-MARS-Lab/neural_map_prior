import os
import pickle as pkl
import tempfile
from copy import deepcopy
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmdet.datasets import DATASETS
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox import get_box_type, LiDARInstance3DBoxes
from .base_dataset import BaseMapDataset
# eval
from .evaluation.hdmap_eval import get_val_info
from .evaluation.precision_recall.average_precision_det import eval_map as eval_map_det
# inherit from LiQi's code
from .nuscences_utils.hdmapnet_data import RaseterizedData
from ..map_tiles.lane_render import \
    load_nusc_data_cities, \
    load_nusc_data_infos, \
    get_token2map_index, \
    split_nusc_geo_bs, \
    resample_by_geo, \
    map_geo_loc

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

eval_cfg = dict(
    raster=False,
    patch_size=(30, 60),
    evaluation_cfg=dict(
        version='v1.0-trainval',
        result_path='./',
        dataroot='/mnt/datasets/nuScenes/',
        eval_set='val',
        num_class=3,
        bsz=4,
        CD_threshold=5,
        thickness=5,
        xbound=[-30.0, 30.0, 0.15],
        ybound=[-15.0, 15.0, 0.15],
        class_names=[
            'ped_crossing',  # 0
            'road_divider', 'lane_divider',  # 1
            'contours',  # 3
            'others',  # -1
        ],
    )
)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles'),
    # dict(type='ResizeMultiViewImages',
    #     img_scale=(128, 352),
    #     ),
    dict(
        type='VectorizeLocalMap',
        data_root='/mnt/datasets/nuScenes/',
        patch_size=(30, 60),
        sample_dist=0.7,
        num_samples=150,
        padding=True,
        normalize=False,
    ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='Pad3D', size_divisor=32),
    dict(
        # type='DefaultFormatBundle3D',
        type='FormatBundleMap',
        # class_names=class_names,
        with_gt=False),
    dict(type='Collect3D', keys=['img', 'vectors'], meta_keys=(
        'img_filenames', 'ori_shape', 'img_shape', 'lidar2img',
        'pad_shape', 'scale_factor', 'flip', 'cam_intrinsics',
        'img_norm_cfg', 'sample_idx',
        'cam2ego_rotations', 'cam2ego_translations',
        'ego2global_translation', 'ego2global_rotation'))
]
MAPS = ['boston-seaport', 'singapore-hollandvillage',
        'singapore-onenorth', 'singapore-queenstown']


@DATASETS.register_module(force=True)
class nuScenesMapDataset(BaseMapDataset):

    def __init__(self,
                 data_root,
                 ann_file,
                 modality=dict(
                     use_camera=True,
                     use_lidar=False,
                     use_radar=False,
                     use_map=True,
                     use_external=False,
                 ),
                 pipeline=None,
                 eval_cfg=dict(
                     raster=False,
                     patch_size=(30, 60),
                 ),
                 rasterized_cfg=None,
                 load_interval=1,
                 classes=None,
                 test_mode=False,
                 samples_per_gpu=1,
                 num_class=None,
                 load_bbox=False,
                 box_type_3d='LiDAR',
                 obj_class_name=None,
                 class_name=None,
                 work_dir='',
                 sort_train_infos=False,
                 sort_val_infos=False,
                 sort_batch_size=None,
                 gm_grid_size=None,
                 origin_sampler=False,
                 nusc_city_split=False,
                 nusc_new_split=False,
                 **kwargs):

        self.origin_sampler = origin_sampler
        self.nusc_city_split = nusc_city_split
        self.nusc_new_split = nusc_new_split

        if sort_train_infos:
            self.sort_train_infos = sort_train_infos
            self.sort_batch_size = sort_batch_size
            self.gm_grid_size = gm_grid_size
        if sort_val_infos:
            self.sort_val_infos = sort_val_infos
            self.sort_batch_size = sort_batch_size
            self.gm_grid_size = gm_grid_size

        # 3dbbox args
        self.load_bbox = load_bbox
        if load_bbox:
            self.with_velocity = False
            self.ego_pose = True
            self.Obj_CLASSES = tuple(obj_class_name)
            self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
            self.use_valid_flag = False

        super().__init__(
            data_root,
            ann_file,
            modality=modality,
            pipeline=pipeline,
            load_interval=load_interval,
            classes=None,
            test_mode=False
        )

        self.eval_cfg = eval_cfg

        if rasterized_cfg is not None:
            self.use_rasterized = True
            rasterized_cfg.raw_dataset_cfg.version = self.version
            rasterized_cfg.raw_dataset_cfg.data_root = data_root
            self.get_rasterized_data = RaseterizedData(**rasterized_cfg)
        else:
            self.use_rasterized = False

        self.num_class = num_class
        self.class_name = class_name

        self.ann_file = ann_file
        self.work_dir = work_dir

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        if self.load_bbox:
            results['bbox3d_fields'] = []
            results['pts_mask_fields'] = []
            results['pts_seg_fields'] = []
            results['bbox_fields'] = []
            results['mask_fields'] = []
            results['seg_fields'] = []
            results['box_type_3d'] = self.box_type_3d
            results['box_mode_3d'] = self.box_mode_3d

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        local_rank = torch.distributed.get_rank()
        return np.random.choice(self.gpu2sample_id[local_rank])
        # pool = np.where(self.flag == self.flag[idx])[0]
        # return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            line_num = len(data['vectors']) if 'vectors' in data else 1
            if data is None or line_num == 0:
                if self.origin_sampler:
                    idx = super()._rand_another(idx)
                else:
                    idx = self._rand_another(idx)
                continue
            return data

    def gen_token2map_info(self, split):
        data_infos = load_nusc_data_infos(split)
        data_cities = load_nusc_data_cities(split)

        if hasattr(self, 'sort_batch_size') and not (self.sort_batch_size == 1):
            tail = len(data_infos) % self.sort_batch_size
            data_infos = data_infos[:-tail]
            data_cities = data_cities[:-tail]
            assert len(data_infos) == len(data_cities), \
                (len(data_infos), len(data_cities))

        return data_infos, data_cities

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if (hasattr(self, 'data_info_boston') and self.data_info_boston) or \
                (hasattr(self, 'nusc_city_split') and self.nusc_city_split) or \
                hasattr(self, 'nusc_new_split') and self.nusc_new_split:
            temp_ann_file = ['/public/MARS/datasets/nuScenes/nuScences_map_trainval_infos_train.pkl',
                             '/public/MARS/datasets/nuScenes/nuScences_map_trainval_infos_val.pkl']
            data = [mmcv.load(ann) for ann in temp_ann_file]
            data_infos = list(data[0]['infos']) + list(data[1]['infos'])
        else:
            data = mmcv.load(ann_file)
            data_infos = list(data['infos'])

        if 'infos_train' in ann_file and not hasattr(self, 'sort_train_infos'):
            if hasattr(self, 'nusc_city_split') and self.nusc_city_split:
                new_train_infos = []
                new_train_data_cities = []
                train_infos, train_data_cities = self.gen_token2map_info('train')
                val_infos, val_data_cities = self.gen_token2map_info('val')
                data_infos = train_infos + val_infos
                data_cities = train_data_cities + val_data_cities
                for info, city_name in zip(data_infos, data_cities):
                    if city_name == 'singapore-onenorth' or \
                            city_name == 'boston-seaport':
                        new_train_infos.append(deepcopy(info))
                        new_train_data_cities.append(city_name)
                data_infos = new_train_infos
                data_city_names = new_train_data_cities

                print('dataset train data_infos', len(data_infos))
                print('dataset train data_city_names', len(data_city_names))

            elif hasattr(self, 'nusc_new_split') and self.nusc_new_split:
                new_train_infos = []
                new_train_data_cities = []
                train_infos, train_data_cities = self.gen_token2map_info('train')
                val_infos, val_data_cities = self.gen_token2map_info('val')
                data_infos = train_infos + val_infos
                data_cities = train_data_cities + val_data_cities
                with open(
                        f'/oldhome/xiongx/repository/neural_map_prior_code/project/train_sample_tokens.pkl', 'rb') as f:
                    sample_tokens_train = pkl.load(f)
                for info, city_name in zip(data_infos, data_cities):
                    if info['token'] in sample_tokens_train:
                        new_train_infos.append(deepcopy(info))
                        new_train_data_cities.append(city_name)
                data_infos = new_train_infos
                data_city_names = new_train_data_cities

                print('dataset train data_infos', len(data_infos))
                print('dataset train data_city_names', len(data_city_names))

        if 'infos_val' in ann_file and not hasattr(self, 'sort_val_infos'):
            if hasattr(self, 'nusc_city_split') and self.nusc_city_split:
                new_val_infos = []
                new_val_data_cities = []
                train_infos, train_data_cities = self.gen_token2map_info('train')
                val_infos, val_data_cities = self.gen_token2map_info('val')
                data_infos = train_infos + val_infos
                data_cities = train_data_cities + val_data_cities
                for info, city_name in zip(data_infos, data_cities):
                    if city_name == 'singapore-queenstown' or \
                            city_name == 'singapore-hollandvillage':
                        new_val_infos.append(deepcopy(info))
                        new_val_data_cities.append(city_name)
                data_infos = new_val_infos
                data_city_names = new_val_data_cities

                print('dataset val data_infos', len(data_infos))
                print('dataset val data_city_names', len(data_city_names))

            elif hasattr(self, 'nusc_new_split') and self.nusc_new_split:
                new_val_infos = []
                new_val_data_cities = []
                train_infos, train_data_cities = self.gen_token2map_info('train')
                val_infos, val_data_cities = self.gen_token2map_info('val')
                data_infos = train_infos + val_infos
                data_cities = train_data_cities + val_data_cities
                with open(
                        f'/oldhome/xiongx/repository/neural_map_prior_code/project/val_sample_tokens.pkl', 'rb') as f:
                    sample_tokens_val = pkl.load(f)
                for info, city_name in zip(data_infos, data_cities):
                    if info['token'] in sample_tokens_val:
                        new_val_infos.append(deepcopy(info))
                        new_val_data_cities.append(city_name)
                data_infos = new_val_infos
                data_city_names = new_val_data_cities

                print('dataset val data_infos', len(data_infos))
                print('dataset val data_city_names', len(data_city_names))

        if 'infos_train' in ann_file:
            dataset = 'train'
            if hasattr(self, 'sort_train_infos') and self.sort_train_infos:
                if hasattr(self, 'nusc_city_split') and self.nusc_city_split:
                    new_train_infos = []
                    new_train_data_cities = []
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = train_infos + val_infos
                    data_cities = train_data_cities + val_data_cities
                    for info, city_name in zip(data_infos, data_cities):
                        if city_name == 'singapore-onenorth' or \
                                city_name == 'boston-seaport':
                            new_train_infos.append(deepcopy(info))
                            new_train_data_cities.append(city_name)
                    data_infos = new_train_infos
                    data_city_names = new_train_data_cities

                elif hasattr(self, 'nusc_new_split') and self.nusc_new_split:
                    new_train_infos = []
                    new_train_data_cities = []
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = train_infos + val_infos
                    data_cities = train_data_cities + val_data_cities
                    with open(
                            f'/oldhome/xiongx/repository/neural_map_prior_code/project/train_sample_tokens.pkl',
                            'rb') as f:
                        sample_tokens_train = pkl.load(f)
                    for info, city_name in zip(data_infos, data_cities):
                        if info['token'] in sample_tokens_train:
                            new_train_infos.append(deepcopy(info))
                            new_train_data_cities.append(city_name)
                    data_infos = new_train_infos
                    data_city_names = new_train_data_cities

                else:
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    data_infos = train_infos
                    data_city_names = train_data_cities

                print('dataset train data_infos', len(data_infos))
                print('dataset train data_city_names', len(data_city_names))

                if (hasattr(self, 'nusc_city_split') and self.nusc_city_split) or (
                        hasattr(self, 'nusc_new_split') and self.nusc_new_split):
                    dataset = 'city'
                else:
                    dataset = 'train'
                nusc_train_min_geo_loc, nusc_train_max_geo_loc = map_geo_loc(dataset)
                train_token2map_index, train_map_index2token = \
                    get_token2map_index(
                        data_infos,
                        data_city_names,
                        nusc_train_min_geo_loc,
                        nusc_train_max_geo_loc,
                        map_prior_tile_size=self.gm_grid_size
                    )
                train_gpu2city = split_nusc_geo_bs(
                    self.sort_batch_size,
                    data_infos,
                    data_city_names,
                    dataset,
                    train_token2map_index,
                    self.gm_grid_size,
                    train_map_index2token)
                self.gpu2sample_id = resample_by_geo(
                    self.sort_batch_size,
                    data_infos,
                    data_city_names,
                    dataset,
                    train_gpu2city,
                    train_token2map_index
                )

        elif 'infos_val' in ann_file:
            dataset = 'val'
            if hasattr(self, 'sort_val_infos') and self.sort_val_infos:
                if hasattr(self, 'nusc_city_split') and self.nusc_city_split:
                    new_val_infos = []
                    new_val_data_cities = []
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = train_infos + val_infos
                    data_cities = train_data_cities + val_data_cities
                    for info, city_name in zip(data_infos, data_cities):
                        if city_name == 'singapore-queenstown' or \
                                city_name == 'singapore-hollandvillage':
                            new_val_infos.append(deepcopy(info))
                            new_val_data_cities.append(city_name)
                    data_infos = new_val_infos
                    data_city_names = new_val_data_cities

                elif hasattr(self, 'nusc_new_split') and self.nusc_new_split:
                    new_val_infos = []
                    new_val_data_cities = []
                    train_infos, train_data_cities = self.gen_token2map_info('train')
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = train_infos + val_infos
                    data_cities = train_data_cities + val_data_cities
                    with open(
                            f'/oldhome/xiongx/repository/neural_map_prior_code/project/val_sample_tokens.pkl',
                            'rb') as f:
                        sample_tokens_val = pkl.load(f)
                    for info, city_name in zip(data_infos, data_cities):
                        if info['token'] in sample_tokens_val:
                            new_val_infos.append(deepcopy(info))
                            new_val_data_cities.append(city_name)
                    data_infos = new_val_infos
                    data_city_names = new_val_data_cities

                else:
                    val_infos, val_data_cities = self.gen_token2map_info('val')
                    data_infos = val_infos
                    data_city_names = val_data_cities

                print('dataset val data_infos', len(data_infos))
                print('dataset val data_city_names', len(data_city_names))

                if (hasattr(self, 'nusc_city_split') and self.nusc_city_split) or (
                        hasattr(self, 'nusc_new_split') and self.nusc_new_split):
                    dataset = 'city'
                else:
                    dataset = 'val'
                nusc_val_min_geo_loc, nusc_val_max_geo_loc = map_geo_loc(dataset)
                val_token2map_index, val_map_index2token = \
                    get_token2map_index(
                        data_infos,
                        data_city_names,
                        nusc_val_min_geo_loc,
                        nusc_val_max_geo_loc,
                        map_prior_tile_size=self.gm_grid_size
                    )
                val_gpu2city = split_nusc_geo_bs(
                    self.sort_batch_size,
                    data_infos,
                    data_city_names,
                    dataset,
                    val_token2map_index,
                    self.gm_grid_size,
                    val_map_index2token
                )
                self.gpu2sample_id = resample_by_geo(
                    self.sort_batch_size,
                    data_infos,
                    data_city_names,
                    dataset,
                    val_gpu2city,
                    val_token2map_index
                )
        else:
            dataset = None

        data_infos = data_infos[::self.load_interval]
        if (hasattr(self, 'data_info_boston') and self.data_info_boston) or \
                (hasattr(self, 'nusc_city_split') and self.nusc_city_split) or \
                (hasattr(self, 'nusc_new_split') and self.nusc_new_split):
            self.metadata = data[0]['metadata']
        else:
            self.metadata = data['metadata']
        self.version = self.metadata['version']

        # valid_idx = [21543,  5674,  25126,  24507,  23106,  24691,  21104,
        # 11772,  4919,  21767,  15223,  7287,  24798,  21035,
        # 11482,  25896,  10438,  27207,  17298,  6359,  24478,
        # 17351,  26961,  4195,  3648,  441,  13625,  13574,
        # 25287,  10987,  3696,  5722,  18304,  5029,  15678,
        # 2615,  11722,  26935,  22717,  17697,  23099,  6766,
        # 28064,  214,  10617,  23048,  19420,  13507, 2717,
        # 864,  21289,  17333,  6357,  22966,  3958,  10892, ]

        # if ann_file=='/public/MARS/datasets/nuScenes/nuScences_map_trainval_infos_train.pkl':
        #     data_infos_out = []
        #     for i in valid_idx:
        #         data_infos_out.append(data_infos[i])
        #     return data_infos_out
        # else:
        #     return data_infos
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]

        input_dict = dict(
            sample_idx=info['token'],
            # sample_idx=index,
            timestamp=info['timestamp'] / 1e6,
            lidar2ego_r=info['lidar2ego_rotation'],
            lidar2ego_t=info['lidar2ego_translation'],
            location=info['location']
        )

        # follow liqi's code
        pos_rotation = Quaternion(info['ego2global_rotation'])
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        input_dict.update({'yaw_pitch_roll': yaw_pitch_roll})

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            ego2cam_rts = []
            cam2ego_rots = []
            cam2ego_trans = []
            cam_intrins = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                cam2ego_trans.append(cam_info['sensor2ego_translation'])
                cam2ego_rots.append(Quaternion(
                    cam_info['sensor2ego_rotation']).rotation_matrix)
                cam_intrins.append(cam_info['cam_intrinsic'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                # obtain cam to ego transformation matrix
                ego2cam_r = np.linalg.inv(Quaternion(
                    cam_info['sensor2ego_rotation']).rotation_matrix)
                ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
                ego2cam_rt = np.eye(4)
                ego2cam_rt[:3, :3] = ego2cam_r.T
                ego2cam_rt[3, :3] = -ego2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2cam_rt = (viewpad @ ego2cam_rt.T)
                ego2cam_rts.append(ego2cam_rt)

            input_dict.update(
                dict(
                    img_filenames=image_paths,
                    lidar2img=lidar2img_rts,
                    ego2img=ego2cam_rts,
                    cam2ego_rotations=cam2ego_rots,
                    cam2ego_translations=cam2ego_trans,
                    cam_intrinsics=cam_intrins,
                ))

            if self.use_rasterized:
                pose_record = {
                    'translation': info['ego2global_translation'],
                    'rotation': info['ego2global_rotation']}
                seg_map, inst_mask, direction_mask = \
                    self.get_rasterized_data(pose_record, info['location'])
                input_dict.update(dict(
                    rasterized_gt=dict(
                        seg_map=seg_map,
                        inst_mask=inst_mask,
                        direction_mask=direction_mask,
                    )
                )
                )

        if self.modality['use_lidar']:
            input_dict.update(
                dict(
                    pts_filename=info['lidar_path'],
                    sweeps=info['sweeps'],
                )
            )

        input_dict['location'] = info['location']
        input_dict['ego2global_translation'] = info['ego2global_translation']
        input_dict['ego2global_rotation'] = info['ego2global_rotation']

        if self.load_bbox:
            input_dict['ann_info'] = self.get_ann_info(index)

        return input_dict

    def get_map_info(self, index):
        info = self.data_infos[index]

        map_filename = info['map_path']

        map_filename = os.path.join(self.data_root, map_filename)

        map_ann = np.load(map_filename)
        # maybe do something here
        # or do something in pipeline
        return map_ann

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.
            ego_pose since bbox default is lidar pose

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.Obj_CLASSES:
                gt_labels_3d.append(self.Obj_CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        if self.ego_pose:
            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            gt_bboxes_3d[:, :3] = gt_bboxes_3d[:, :3] @ l2e_r_mat.T
            gt_bboxes_3d[:, :3] = gt_bboxes_3d[:, :3] + np.array(l2e_t)

            rot_sin = l2e_r_mat.T[1, 0]
            rot_cos = l2e_r_mat.T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            gt_bboxes_3d[:, -1:] = gt_bboxes_3d[:, -1:] + angle

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def format_bbox(self, det_res: dict):
        '''
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
            cls1_det: numpy array of shape (n, bbox_size+1) the last element is scores
        annotations (dict): Ground truth annotations. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, npts)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        '''

        preds = det_res
        res = []
        bbox_size = preds['bboxes'].shape[1]

        # process prediction
        for i in range(self.num_class):

            if np.sum(preds['labels'] == i) == 0:
                res.append(np.zeros((0, bbox_size + 1)))
                continue

            selected_idx = preds['labels'] == i

            _det = np.concatenate(
                [preds['bboxes'][selected_idx],
                 preds['scores'][selected_idx][:, None]], axis=-1)
            res.append(_det)

        # for GT
        if det_res['det_gt'] is not None:
            gts = det_res['det_gt']
            ann = {
                'labels': gts['labels'],
                'bboxes': gts['bboxes'],
                'bbox_masks': gts['bbox_masks'], }

        return res, ann

    def format_results(self, results, jsonfile_prefix=None, name=None, only_det=False):
        '''
            results: List[cases]
            cases = {
                'token': <str>
                'lines': List[line]
                'scores': List[float],
                'labels': List[int],
                'nline': Int
            }

        '''

        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = os.path.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        jsonfile_prefix = os.path.join('./work_dirs', 'results')

        meta = self.modality
        meta.update({'vector': not self.eval_cfg.raster})
        nusc_submissions = {
            "meta": meta,
            "results": {},
            "groundTruth": {},  # for validation
            "det_res": {
                'results': {},
                'annotations': {}
            },  # for det
        }

        patch_size = np.array(self.eval_cfg.patch_size)
        print('I\'m setting the formation up!')
        for case in mmcv.track_iter_progress(results):

            if case is None:
                continue

            if self.eval_cfg.raster:
                # nusc_submissions['gt_maps']
                nusc_submissions['results'][case['token']] = {}
                nusc_submissions['results'][case['token']
                ]['map'] = case['pred_map']
                nusc_submissions['results'][case['token']
                ]['confidence_level'] = [1]
            else:
                '''
                    vectorized_line {
                        "pts":               List[<float, 2>]  -- Ordered points to define the vectorized line.
                        "pts_num":           <int>,            -- Number of points in this line.
                        "type":              <0, 1, 2>         -- Type of the line: 0: ped; 1: divider; 2: boundary
                        "confidence_level":  <float>           -- Confidence level for prediction (used by Average Precision)
                    }
                '''
                if 'bboxes' in case:
                    res, ann = self.format_bbox(case)
                    nusc_submissions['det_res']['results'][case['token']] = res
                    nusc_submissions['det_res']['annotations'][case['token']] = ann

                if only_det:
                    continue

                vector_lines = []
                for i in range(case['nline']):
                    # renormalized line from (-1,1) to patch size (-patch_size/2, patch_size/2)
                    # following 3 lines is besed on the inverse operator from preprocess.
                    size = np.array(
                        [patch_size[1], patch_size[0]]) + 2
                    size = size / 2
                    line = case['lines'][i] * size

                    vector_lines.append({
                        'pts': line,
                        'pts_num': len(case['lines'][i]),
                        'type': case['labels'][i],
                        'confidence_level': case['scores'][i],
                    })

                nusc_submissions['results'][case['token']] = {}
                nusc_submissions['results'][case['token']
                ]['vectors'] = vector_lines

                if 'groundTruth' in case:

                    nusc_submissions['groundTruth'][case['token']] = {}
                    vector_lines = []
                    for i in range(case['groundTruth']['nline']):
                        # renormalized line from (-1,1) to patch size (-patch_size/2, patch_size/2)
                        size = np.array(
                            [patch_size[1], patch_size[0]]) + 2
                        size = size / 2
                        line = case['groundTruth']['lines'][i] * size

                        vector_lines.append({
                            'pts': line,
                            'pts_num': len(case['groundTruth']['lines'][i]),
                            'type': case['groundTruth']['labels'][i],
                            'confidence_level': 1.,
                        })

                    nusc_submissions['groundTruth'][case['token']
                    ]['vectors'] = vector_lines

                if 'aux_map' in case.keys():
                    nusc_submissions['results'][case['token']
                    ]['aux_map'] = case['aux_map']

        print('Done!')
        mmcv.mkdir_or_exist(jsonfile_prefix)
        mmcv.mkdir_or_exist(self.work_dir)
        res_path = os.path.join(self.work_dir, '{}.pkl'.format(name))
        mmcv.dump(nusc_submissions, res_path)
        print('Haiya, the results have written in', res_path)

        return res_path

    def evaluate(self,
                 results,
                 logger=None,
                 show=False,
                 online_evaluation=False,
                 name=None,
                 only_det=False,
                 no_det=False,
                 **kwargs):
        '''
        Args:
            results (list[dict]): List of results.

        Returns:
            dict: Evaluation results.
        '''
        name = 'results_nuscence' if name is None else name

        print('len of the results', len(results))
        res_path = self.format_results(results, name=name, only_det=only_det)
        self.eval_cfg.evaluation_cfg['result_path'] = res_path
        self.eval_cfg.evaluation_cfg['ann_file'] = self.ann_file

        if not no_det:
            print('det\'s performance:')
            det_mean_ap, det_map_results = \
                eval_map_det(self.eval_cfg.evaluation_cfg, logger)

            if only_det:
                return {
                    'det_map': mean_ap
                }
        else:
            det_mean_ap = 0

        print('gen\'s performance:')
        # gen_mean_ap, gen_map_results =\
        #    eval_map_gen(self.eval_cfg.evaluation_cfg, True, logger)

        result_dict = get_val_info(self.eval_cfg.evaluation_cfg)
        print('HdmapNet Evaluation Results:')
        pprint(result_dict)

        # print('VectormapNet Evaluation Results:')
        # result_dict.update({
        #    'det_map': det_mean_ap,
        #    'gen_map': gen_mean_ap,
        # })

        # pprint(vec_res_dict)

        return result_dict


def project_labels_to_pic(dataset, idx):
    def int_coords(coords):
        return (int(coords[0]), int(coords[1]))

    def get_proj_mat(intrins, rots, trans):
        K = np.eye(4)
        K[:3, :3] = intrins
        R = np.eye(4)
        R[:3, :3] = rots.transpose(-1, -2)
        T = np.eye(4)
        T[:3, 3] = -trans
        RT = R @ T
        return K @ RT

    def perspective(cam_coords, proj_mat):
        """
        pix_coords = proj_mat @ (x, y, z, 1)
        Project cam2pixel

        Args:
            cam_coords:         [4, npoints]
            proj_mat:           [4, 4]

        Returns:
            pix coords:         [npoints, 2]
        """
        pix_coords = proj_mat @ cam_coords
        valid_idx = pix_coords[2, :] > 0
        pix_coords = pix_coords[:, valid_idx]
        pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
        pix_coords = pix_coords.transpose(1, 0)
        return pix_coords

    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    CAMS = [
        'FRONT',
        'FRONT_RIGHT',
        'FRONT_LEFT',
        'BACK',
        'BACK_LEFT',
        'BACK_RIGHT',
    ]
    data = dataset[idx]
    os.makedirs(
        './plugin/hdmapnet_vectorized/datasets/demos/{}/'.format(idx), exist_ok=True)
    vectors = data['vectors']
    imgs = data['img']
    for i in range(6):
        cam2ego_rots = data['cam2ego_rotations'][i]
        cam2ego_trans = np.array(data['cam2ego_translations'][i])
        cam_intrins = data['cam_intrinsics'][i]

        img = imgs[i]
        P = get_proj_mat(cam_intrins, cam2ego_rots, cam2ego_trans)
        for line in vectors:
            length = line[1]
            pts = line[0][:length]
            label = line[2]

            zeros = np.zeros((length, 1))
            ones = np.ones((length, 1))

            world_coords = np.concatenate(
                [pts, zeros, ones], axis=1).transpose(1, 0)

            pix_coords = perspective(world_coords, P)
            for _, pts in enumerate(pix_coords):
                cv2.circle(img, int_coords(pts), 2, colors[label], 2)
        cv2.imwrite(
            './plugin/hdmapnet_vectorized/datasets/demos/{}/{}.jpg'.format(idx, CAMS[i]), img)


def visdata(dataset, idx):
    data = dataset[idx]
    os.makedirs(
        './plugin/hdmapnet_vectorized/datasets/demos/{}/'.format(idx), exist_ok=True)
    map_mask = np.ones((200, 400, 3), dtype=np.uint8) * 255
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
    colors_plt = ['r', 'b', 'g']
    vectors = data['vectors']
    plt.figure(figsize=(4, 2))
    plt.xlim(-30, 30)
    plt.ylim(-15, 15)
    plt.axis('off')
    for vect, length, label in vectors:
        vect = vect[:length]
        x = np.array([pts[0] for pts in vect])
        y = np.array([pts[1] for pts in vect])
        plt.quiver(0, 0, 2, 0, scale_units='xy',
                   angles='xy', scale=1, color='r')
        plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy',
                   angles='xy', scale=1, color=colors_plt[label])
        plt.savefig('./plugin/hdmapnet_vectorized/datasets/demos/{}/map_new.jpg'.format(idx),
                    bbox_inches='tight', dpi=400)

    plt.savefig('./plugin/hdmapnet_vectorized/datasets/demos/{}/map_new.jpg'.format(idx),
                bbox_inches='tight', dpi=400)

    imgs = data['img']
    cv2.imwrite(
        './plugin/hdmapnet_vectorized/datasets/demos/{}/front.jpg'.format(idx), imgs[0])
    cv2.imwrite(
        './plugin/hdmapnet_vectorized/datasets/demos/{}/front_right.jpg'.format(idx), imgs[1])
    cv2.imwrite(
        './plugin/hdmapnet_vectorized/datasets/demos/{}/front_left.jpg'.format(idx), imgs[2])
    cv2.imwrite(
        './plugin/hdmapnet_vectorized/datasets/demos/{}/back.jpg'.format(idx), imgs[3])
    cv2.imwrite(
        './plugin/hdmapnet_vectorized/datasets/demos/{}/back_left.jpg'.format(idx), imgs[4])
    cv2.imwrite(
        './plugin/hdmapnet_vectorized/datasets/demos/{}/back_right.jpg'.format(idx), imgs[5])


def _test():
    my_dataset = nuScenesMapDataset(
        data_root='/mnt/datasets/nuScenes',
        ann_file='/mnt/datasets/nuScenes/nuScences_map_trainval_infos_val.pkl',
        pipeline=train_pipeline,
        eval_cfg=eval_cfg,
        modality=input_modality,
    )

    # data = my_dataset[0]
    # embed()
    # for i in [0, 15, 50, 100, 150, 200]:
    #     project_labels_to_pic(my_dataset, i)
    #     # visdata(my_dataset, i)
    from mmcv import Config
    cfg = Config.fromfile('./plugin/hdmapnet_vectorized/configs/ipm_detr.py')
    results = mmcv.load('./old_IpmDetr_results.pkl')
    from mmdet3d.datasets import build_dataset
    dataset = build_dataset(cfg.data.test)
    dataset.evaluate(results)


if __name__ == '__main__':
    _test()

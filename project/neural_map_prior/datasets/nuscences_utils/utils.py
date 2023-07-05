"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx, alpha_poly=0.6, alpha_line=1.):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane', 'ped_crossing']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)

    for la in lmap['road_segment']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 0], pts[:, 1], c=(124. / 255., 179. / 255., 210. / 255.), alpha=alpha_poly, linewidth=5)
        # plt.plot(pts[:, 0], pts[:, 1], c=(0., 1., 0.), alpha=alpha_poly, linewidth=5)
    for la in lmap['lane']:
        pts = (la - bx) / dx
        plt.fill(pts[:, 0], pts[:, 1], c=(74. / 255., 163. / 255., 120. / 255.), alpha=alpha_poly)
    for la in lmap['ped_crossing']:
        dist = np.square(la[1:, :] - la[:-1, :]).sum(-1)
        x1, x2 = np.argsort(dist)[-2:]
        pts = (la - bx) / dx
        # plt.plot(pts[:, 0], pts[:, 1], c=(247./255., 129./255., 132./255.), alpha=alpha_poly, linewidth=5)
        plt.plot(pts[x1:x1 + 2, 0], pts[x1:x1 + 2, 1], c=(1., 0., 0.), alpha=alpha_poly, linewidth=5)
        plt.plot(pts[x2:x2 + 2, 0], pts[x2:x2 + 2, 1], c=(1., 0., 0.), alpha=alpha_poly, linewidth=5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        # plt.plot(pts[:, 0], pts[:, 1], c=(159. / 255., 0.0, 1.0), alpha=alpha_line, linewidth=5)
        plt.plot(pts[:, 0], pts[:, 1], c=(0., 0., 1.), alpha=alpha_poly, linewidth=5)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        # plt.plot(pts[:, 0], pts[:, 1], c=(0.0, 0.0, 1.0), alpha=alpha_line, linewidth=5)
        plt.plot(pts[:, 0], pts[:, 1], c=(0., 0., 1.), alpha=alpha_poly, linewidth=5)


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
            )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # img = img.resize((352, 128))
    # post_rot[0, 0] *= 352/1600
    # post_rot[1, 1] *= 128/900

    # return img, post_rot, post_tran

    # adjust image
    img = img.resize(resize_dims)
    # img = img.crop(crop)
    # if flip:
    #     img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    # img = img.rotate(rotate)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot = rot_resize @ post_rot
    post_tran = rot_resize @ post_tran

    # if flip:
    #     rot_flip = torch.Tensor([[-1, 0],
    #                                 [0, 1]])
    #     tran_flip = torch.Tensor([resize_dims[0], 0])
    #     post_rot = rot_flip @ post_rot
    #     post_tran = rot_flip @ post_tran + tran_flip

    # rot_rot = get_rot(rotate / 180 * np.pi)
    # tran_flip = torch.Tensor(resize_dims) / 2
    # post_rot = rot_rot @ post_rot
    # post_tran = rot_rot @ post_tran + tran_flip

    return img, post_rot, post_tran

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0],
                          [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


random_erasing = torchvision.transforms.RandomErasing()

denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))

normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                       map_name=map_name) for map_name in [
                     "singapore-hollandvillage",
                     "singapore-queenstown",
                     "boston-seaport",
                     "singapore-onenorth",
                 ]}
    return nusc_maps


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot


def pad_or_trim_to_np(x, shape, pad_val=0):
    shape = np.asarray(shape)
    pad = shape - np.minimum(np.shape(x), shape)
    zeros = np.zeros_like(pad)
    x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
    return x[:shape[0], :shape[1]]

import pickle as pkl

import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R


def gen_matrix(ego2global_rotation, ego2global_translation):
    rotation_xyz = np.roll(ego2global_rotation, shift=-1)
    trans = np.eye(4)
    trans[:3, 3] = ego2global_translation
    trans[:3, :3] = R.from_quat(rotation_xyz).as_matrix()
    return trans


def city_geo_bound(nusc, data_infos):
    city_names = ['singapore-onenorth',
                  'boston-seaport',
                  'singapore-queenstown',
                  'singapore-hollandvillage']

    city_geo_loc = {city_name: list() for city_name in city_names}

    for info in data_infos:
        scene = nusc.get('scene', info['scene_token'])
        cn = nusc.get('log', scene['log_token'])['location']
        trans = gen_matrix(
            info['ego2global_rotation'],
            info['ego2global_translation'])
        city_geo_loc[cn].append(trans[:3, 3])

    city_geo_loc = {k: np.array(v) for k, v in city_geo_loc.items()}
    max_geo_loc = {k: v.max(0) for k, v in city_geo_loc.items()}
    min_geo_loc = {k: v.min(0) for k, v in city_geo_loc.items()}
    return min_geo_loc, max_geo_loc


with open('/datasets/nuScenes/nuscenes_infos_temporal_train.pkl', 'rb') as f:
    train_infos = pkl.load(f)['infos']
print('load training data infos for nuscenes dataset...')

with open('/datasets/nuScenes/nuscenes_infos_temporal_val.pkl', 'rb') as f:
    val_infos = pkl.load(f)['infos']
print('load val data infos for nuscenes dataset...')

nusc = NuScenes(version='v1.0-trainval', dataroot='/public/MARS/datasets/nuScenes/', verbose=True)

print(city_geo_bound(nusc, train_infos))
print(city_geo_bound(nusc, val_infos))

train_min_geo_loc = {'singapore-onenorth': np.array([118.5702648, 420.87564563, 0.]),
                     'boston-seaport': np.array([298.10073396, 328.68013429, 0.]),
                     'singapore-queenstown': np.array([347.6363536, 862.96163218, 0.]),
                     'singapore-hollandvillage': np.array([442.68427172, 902.72119429, 0.])}
train_max_geo_loc = {'singapore-onenorth': np.array([1231.63951313, 1776.71524764, 0.]),
                     'boston-seaport': np.array([2526.75659041, 1895.6798286, 0.]),
                     'singapore-queenstown': np.array([2685.46302851, 3297.92406389, 0.]),
                     'singapore-hollandvillage': np.array([2489.16021845, 2838.52768129, 0.])}
val_min_geo_loc = {'singapore-onenorth': np.array([118.97422868, 408.42224536, 0.]),
                   'boston-seaport': np.array([412.47323101, 555.56404621, 0.]),
                   'singapore-queenstown': np.array([524.53205984, 871.62667194, 0.]),
                   'singapore-hollandvillage': np.array([608.50758612, 2007.31648805, 0.])}
val_max_geo_loc = {'singapore-onenorth': np.array([1231.66762219, 1731.31826411, 0.]),
                   'boston-seaport': np.array([2366.80451678, 1719.72736379, 0.]),
                   'singapore-queenstown': np.array([2043.40095771, 3332.8954411, 0.]),
                   'singapore-hollandvillage': np.array([2459.24878815, 2835.49775344, 0.])}

import copy
import glob
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.nn import functional as F
from tqdm import tqdm

bev_real_h = 30
bev_real_w = 60
bev_radius = np.sqrt(bev_real_h ** 2 + bev_real_w ** 2) / 2
train_min_geo_loc = {'singapore-onenorth': np.array([118., 420.]) - bev_radius,
                     'boston-seaport': np.array([298., 328.]) - bev_radius,
                     'singapore-queenstown': np.array([347., 862.]) - bev_radius,
                     'singapore-hollandvillage': np.array([442., 902.]) - bev_radius}
train_max_geo_loc = {'singapore-onenorth': np.array([1232., 1777.]) + bev_radius,
                     'boston-seaport': np.array([2527., 1896.]) + bev_radius,
                     'singapore-queenstown': np.array([2686., 3298.]) + bev_radius,
                     'singapore-hollandvillage': np.array([2490., 2839.]) + bev_radius}
val_min_geo_loc = {'singapore-onenorth': np.array([118., 408.]) - bev_radius,
                   'boston-seaport': np.array([412., 555.]) - bev_radius,
                   'singapore-queenstown': np.array([524., 871.]) - bev_radius,
                   'singapore-hollandvillage': np.array([608., 2007.]) - bev_radius}
val_max_geo_loc = {'singapore-onenorth': np.array([1232., 1732.]) + bev_radius,
                   'boston-seaport': np.array([2367., 1720.]) + bev_radius,
                   'singapore-queenstown': np.array([2044., 3333.]) + bev_radius,
                   'singapore-hollandvillage': np.array([2460., 2836.]) + bev_radius}
#
# val_min_geo_loc = {'singapore-onenorth': np.array([118., 420.]) - bev_radius,
#                    'boston-seaport': np.array([298., 328.]) - bev_radius,
#                    'singapore-queenstown': np.array([347., 862.]) - bev_radius,
#                    'singapore-hollandvillage': np.array([442., 902.]) - bev_radius}
# val_max_geo_loc = {'singapore-onenorth': np.array([1232., 1777.]) + bev_radius,
#                    'boston-seaport': np.array([2527., 1896.]) + bev_radius,
#                    'singapore-queenstown': np.array([2686., 3298.]) + bev_radius,
#                    'singapore-hollandvillage': np.array([2490., 2839.]) + bev_radius}

city_min_geo_loc = {}
for city, geo_min in train_min_geo_loc.items():
    city_min_geo_loc[city] = np.min(np.stack([val_min_geo_loc[city], geo_min]), axis=0)
city_max_geo_loc = {}
for city, geo_max in train_max_geo_loc.items():
    city_max_geo_loc[city] = np.max(np.stack([val_max_geo_loc[city], geo_max]), axis=0)


def map_path(map_attribute, dataset, prefix_name='map', gpu_name=None):
    if 'prefix' in map_attribute:
        prefix_name = map_attribute['prefix']
    dir_name = '_'.join([prefix_name, map_attribute['tile_param']['data_type'],
                         str(map_attribute['global_map_tile_size'][0]),
                         str(map_attribute['global_map_tile_size'][1])])
    if gpu_name is not None:
        return os.path.join(map_attribute['root_dir'], dir_name, dataset, gpu_name)
    return os.path.join(map_attribute['root_dir'], dir_name, dataset)


def get_bev_coords(bev_bound_w, bev_bound_h, bev_w, bev_h, type_flag='numpy'):
    '''
    Args:
        bev_bound_w (tuple:2):
        bev_bound_h (tuple:2):
        bev_w (int):
        bev_h (int):

    Returns: (bev_h, bev_w, 4)

    '''

    if type_flag == 'torch':
        sample_coords = torch.stack(torch.meshgrid(
            torch.linspace(bev_bound_w[0], bev_bound_w[1], int(bev_w), dtype=torch.float32),
            torch.linspace(bev_bound_h[0], bev_bound_h[1], int(bev_h), dtype=torch.float32)
        ), axis=2).transpose(1, 0)
        assert sample_coords.shape[0] == bev_h, sample_coords.shape[1] == bev_w
        zeros = torch.zeros((bev_h, bev_w, 1), dtype=sample_coords.dtype)
        ones = torch.ones((bev_h, bev_w, 1), dtype=sample_coords.dtype)
        sample_coords = torch.cat([sample_coords, zeros, ones], dim=-1)
    elif type_flag == 'numpy':
        sample_coords = np.meshgrid(np.linspace(bev_bound_w[0], bev_bound_w[1], int(bev_w), dtype=np.float32),
                                    np.linspace(bev_bound_h[0], bev_bound_h[1], int(bev_h), dtype=np.float32))
        sample_coords = np.stack(sample_coords, axis=2)
        assert sample_coords.shape[0] == bev_h, sample_coords.shape[1] == bev_w
        zeros = np.zeros((bev_h, bev_w, 1), dtype=sample_coords.dtype)
        ones = np.ones((bev_h, bev_w, 1), dtype=sample_coords.dtype)
        sample_coords = np.concatenate([sample_coords, zeros, ones], axis=-1)
    return sample_coords


def get_coords_resample(bev_feature,
                        pad_bev_feature_int,
                        pad_bev_feature,
                        ego2ego,
                        global_bound_h,
                        global_bound_w,
                        real_h=30,
                        real_w=60,
                        bev_h=200,
                        bev_w=400,
                        reverse=False,
                        flag='replace',
                        traversal_id=None,
                        global_factor=1,
                        bev_feature_int=None,
                        view_info=False,
                        type_flag='numpy'):
    '''
    Args:
        bev_feature (B, bev_h, bev_w, C):
        pad_bev_feature (B, bev_h, bev_w, C):
        ego2ego (4, 4):
        real_h (int):
        real_w (int):
    Returns: (B, bev_h, bev_w, C)

    '''
    # device = bev_feature.device
    # B, bev_h, bev_w, C = bev_feature.size()

    bev_bound_h, bev_bound_w = \
        [(-row[0] / 2 + row[0] / row[1] / 2, row[0] / 2 - row[0] / row[1] / 2)
         for row in ((real_h, bev_h), (real_w, bev_w))]
    # bev_bound_h, bev_bound_w = \
    #     [(-row[0] / 2, row[0] / 2)
    #      for row in ((real_h, bev_h), (real_w, bev_w))]
    grid_len_h = real_h / (bev_h / global_factor)
    grid_len_w = real_w / (bev_w / global_factor)

    bev_coords = get_bev_coords(bev_bound_w, bev_bound_h, bev_w, bev_h, type_flag)

    if type_flag == 'torch':
        ego2ego = bev_coords.new_tensor(ego2ego)
        bev_coords = bev_coords.reshape(-1, 4).permute(1, 0)
    elif type_flag == 'numpy':
        bev_coords = bev_coords.reshape(-1, 4).transpose(1, 0)

    trans_bev_coords = ego2ego @ bev_coords

    bev_coord_w = trans_bev_coords[0, :]
    bev_coord_h = trans_bev_coords[1, :]

    if type_flag == 'torch':
        bev_index_w = torch.floor((bev_coord_w - global_bound_w[0]) / grid_len_w).to(torch.int64)
        bev_index_h = torch.floor((bev_coord_h - global_bound_h[0]) / grid_len_h).to(torch.int64)
    elif type_flag == 'numpy':
        bev_index_w = np.floor((bev_coord_w - global_bound_w[0]) / grid_len_w).astype(np.int64)
        bev_index_h = np.floor((bev_coord_h - global_bound_h[0]) / grid_len_h).astype(np.int64)

    bev_index_w = bev_index_w.reshape(bev_h, bev_w)
    bev_index_h = bev_index_h.reshape(bev_h, bev_w)

    bev_coord_mask = \
        (global_bound_w[0] <= bev_coord_w) & (bev_coord_w < global_bound_w[1]) & \
        (global_bound_h[0] <= bev_coord_h) & (bev_coord_h < global_bound_h[1])
    bev_coord_mask = bev_coord_mask.reshape(bev_h, bev_w)

    if type_flag == 'torch':
        index_h, index_w = torch.where(bev_coord_mask.reshape(bev_h, bev_w))
    elif type_flag == 'numpy':
        index_h, index_w = np.where(bev_coord_mask.reshape(bev_h, bev_w))

    # overlap_feats = bev_feature[:, index_h, index_w, :]
    # if type_flag == 'torch':
    #     pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] = overlap_feats.to('cpu')
    # elif type_flag == 'numpy':
    #     pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] = overlap_feats

    if pad_bev_feature_int is not None:
        pad_bev_feature_int[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] += 1
    if reverse:
        overlap_feats = pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :]
        if type_flag == 'torch':
            overlap_feats = overlap_feats.to(bev_feature.device).clone()
        bev_feature[:, index_h, index_w, :] = overlap_feats

        if bev_feature_int is not None:
            bev_feature_int[:, index_h, index_w, :] = \
                pad_bev_feature_int[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :].to(
                    bev_feature_int.device).clone()
    else:
        if pad_bev_feature is not None:
            overlap_feats = bev_feature[:, index_h, index_w, :]
            if type_flag == 'torch':
                overlap_feats = overlap_feats.to('cpu')
            if flag == 'add':
                pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] += overlap_feats
            elif flag == 'replace':
                pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] = overlap_feats


def load_nusc_data_infos(dataset, root=None):
    if root is None:
        root = '~/neural_map_prior/data/nuscenes/'
    with open(f'{root}/nuScences_map_trainval_infos_{dataset}.pkl', 'rb') as f:
        infos = pkl.load(f)['infos']
    print(f'load {len(infos)} {dataset} data infos for nuscenes dataset...')
    return infos


def load_nusc_data_cities(dataset, root=None):
    if root is None:
        root = '~/neural_map_prior/data/nuscenes_infos'
    with open(f'{root}/{dataset}_city_infos.pkl', 'rb') as f:
        data_city_names = pkl.load(f)
    print(f'load {len(data_city_names)} {dataset} city names for nuscenes dataset...')
    return data_city_names


def map_geo_loc(dataset):
    geo_loc = {
        'train':
            (train_min_geo_loc, train_max_geo_loc),
        'val':
            (val_min_geo_loc, val_max_geo_loc),
        'city':
            (city_min_geo_loc, city_max_geo_loc)}
    return geo_loc[dataset]


def gen_matrix(ego2global_rotation, ego2global_translation):
    rotation_xyz = np.roll(ego2global_rotation, shift=-1)
    trans = np.eye(4)
    trans[:3, 3] = ego2global_translation
    trans[:3, :3] = R.from_quat(rotation_xyz).as_matrix()
    return trans


def map_tile_bound(min_geo_loc, max_geo_loc, city_name, map_index, map_prior_tile_size):
    city_map_tile_interval = get_city_map_tile_interval(city_name, min_geo_loc, max_geo_loc, map_prior_tile_size)
    map_slice_min_bound = map_index * city_map_tile_interval + min_geo_loc[city_name]
    map_slice_max_bound = (map_index + 1) * city_map_tile_interval + min_geo_loc[city_name]
    return map_slice_min_bound, map_slice_max_bound


def map_tile_raster_size(tile_intervals, map_prior_raster_size):
    tile_grid_size = tile_intervals / np.array(map_prior_raster_size, np.float32)
    return tile_grid_size


def map_tile_interval(city_min_bound, city_max_bound, map_prior_tile_size):
    city_range = city_max_bound - city_min_bound
    tile_interval = city_range / np.array(map_prior_tile_size, np.int32)
    return tile_interval


def get_city_map_tile_interval(city_name, min_geo_loc, max_geo_loc, map_prior_tile_size):
    city_min_bound = min_geo_loc[city_name]
    city_max_bound = max_geo_loc[city_name]
    city_map_tile_interval = map_tile_interval(city_min_bound, city_max_bound, map_prior_tile_size)
    return city_map_tile_interval


def get_array_data_type(data_type):
    return getattr(np, data_type)


def get_tensor_data_type(data_type):
    return getattr(torch, data_type)


def creat_empty_map_tile_array(map_tile_size, embed_dims, data_type, num_traversals=1):
    '''
    Create numpy array for one map tile.
    Args:
        map_tile_size ():
        embed_dims ():
        data_type ():
        num_traversals ():

    Returns:

    '''
    data_type = get_array_data_type(data_type)
    tile_height = map_tile_size[0]
    tile_width = map_tile_size[1]
    tile_height_ceil = int(np.ceil(tile_height))
    tile_width_ceil = int(np.ceil(tile_width))
    tile_empty_array = np.zeros((
        num_traversals, tile_height_ceil, tile_width_ceil, embed_dims
    ), dtype=data_type)
    return tile_empty_array


def creat_empty_map_tile_tensor(map_tile_size, embed_dims, data_type, num_traversals=1):
    '''
    Create torch tensor for one map tile.
    Args:
        map_tile_size ():
        embed_dims ():
        data_type ():
        num_traversals ():

    Returns:

    '''
    data_type = get_tensor_data_type(data_type)
    tile_height = map_tile_size[0]
    tile_width = map_tile_size[1]
    tile_height_ceil = int(np.ceil(tile_height))
    tile_width_ceil = int(np.ceil(tile_width))
    tile_empty_tensor = torch.zeros((
        num_traversals, tile_height_ceil, tile_width_ceil, embed_dims
    ), dtype=data_type)
    return tile_empty_tensor


def creat_global_map(map_slice_dict, map_attribute, min_geo_loc, max_geo_loc):
    gm_tile_size = map_attribute['global_map_tile_size']
    gm_raster_size = map_attribute['global_map_raster_size']

    for city_name in tqdm(min_geo_loc.keys()):
        city_min_bound = min_geo_loc[city_name]
        city_max_bound = max_geo_loc[city_name]

        tile_interval = map_tile_interval(city_min_bound, city_max_bound, gm_tile_size)
        tile_grid_size = map_tile_raster_size(tile_interval, gm_raster_size)

        for i in range(gm_tile_size[0]):
            for j in range(gm_tile_size[1]):
                map_slice_dict[f'map_{city_name}_{i}_{j}'] = \
                    creat_empty_map_tile_array(tile_grid_size, **map_attribute['tile_param'])


def save_global_map(cus_map_path, map_slice_dict, map_attribute, min_geo_loc, max_geo_loc):
    gm_tile_size = map_attribute['global_map_tile_size']

    print('save map in the path of: ', cus_map_path)
    for city_name in tqdm(min_geo_loc.keys()):
        os.makedirs(os.path.join(cus_map_path, city_name), exist_ok=True)
        for i in range(gm_tile_size[0]):
            for j in range(gm_tile_size[1]):
                torch.save(map_slice_dict[f'map_{city_name}_{i}_{j}'],
                           os.path.join(cus_map_path, city_name, f'map_{i}_{j}.pth'))


def init_city_map_tile_dict(dataset_cities, map_prior_tile_size):
    return \
        {city_name: {f'map_index_{i}_{j}': list()
                     for i in range(map_prior_tile_size[0])
                     for j in range(map_prior_tile_size[1])
                     } for city_name in dataset_cities}


def sample2map_index(info, city_min_bound, tile_interval):
    current_loc = np.array(info['ego2global_translation'][:2])
    map_index = ((current_loc - city_min_bound) / tile_interval).astype(np.int32)
    return map_index


def get_token2map_index(data_infos, city_names,
                        min_geo_loc, max_geo_loc,
                        map_prior_tile_size):
    token2map_index = {}
    dataset_cities = [key for key in min_geo_loc]
    map_index2token = init_city_map_tile_dict(dataset_cities, map_prior_tile_size)

    for i, (info, data_cn) in tqdm(enumerate(zip(data_infos, city_names))):
        city_map_tile_interval = get_city_map_tile_interval(data_cn, min_geo_loc, max_geo_loc, map_prior_tile_size)
        city_min_bound = min_geo_loc[data_cn]
        map_index = sample2map_index(info, city_min_bound, city_map_tile_interval)

        token2map_index.update({info['token']:
            {
                'index': i,
                'num_occ_map': 1,
                'map_index_list': [map_index],
                'loc': data_cn,
                'timestamp': info['timestamp']
            }})
        map_index2token[data_cn][f'map_index_{map_index[0]}_{map_index[1]}'].append(info['token'])
    return token2map_index, map_index2token


def split_nusc_geo_bs(batch_size, data_infos, data_city_names, dataset,
                      token2map_index=None, gm_grid_size=None, map_index2token=None, multi_map_index=False):
    min_geo_loc, max_geo_loc = map_geo_loc(dataset)
    assert len(data_infos) == len(data_city_names) == len(token2map_index)
    num_geo = batch_size
    num_sample_per_geo = len(data_infos) // batch_size

    city_samples = {city: 0 for city in min_geo_loc.keys()}
    for city in data_city_names:
        num_sample = city_samples[city]
        num_sample += 1
        city_samples[city] = num_sample
    city_samples = [{'city_name': k, 'num_sample': v} for k, v in city_samples.items()]
    city_samples = sorted(city_samples, key=lambda e: e['num_sample'])[::-1]
    city_sort_key = [c['city_name'] for c in city_samples]

    city_samples = {city_name: {f'map_{i}_{j}': 0
                                for i in range(gm_grid_size[0])
                                for j in range(gm_grid_size[1])}
                    for city_name in min_geo_loc.keys()}
    for info, city in zip(data_infos, data_city_names):
        map_index = token2map_index[info['token']]['map_index_list'][0]
        num_sample = city_samples[city][f'map_{map_index[0]}_{map_index[1]}']
        num_sample += 1
        city_samples[city][f'map_{map_index[0]}_{map_index[1]}'] = num_sample

    city_samples_sort = []
    for k in city_sort_key:
        per_city_samples = [{'map_index_name': k, 'num_sample': v} for k, v in city_samples[k].items()]
        per_city_samples = sorted(per_city_samples, key=lambda e: e['num_sample'])[::-1]
        city_samples_sort.append({k: per_city_samples})

    gpu2city_map_index = {i: list() for i in range(num_geo)}
    for gpu_id, city_map_index_list in gpu2city_map_index.items():
        get_num_sample = num_sample_per_geo
        for cs in city_samples_sort:
            for city_name, ms in cs.items():
                for sample in ms:
                    num_sample = sample['num_sample']
                    map_index_name = sample['map_index_name']
                    if get_num_sample > 0:
                        if num_sample >= get_num_sample:
                            city_map_index_list.append((f'{city_name}_{map_index_name}', get_num_sample))
                            sample['num_sample'] = num_sample - get_num_sample
                            get_num_sample = 0
                        elif num_sample != 0:
                            city_map_index_list.append((f'{city_name}_{map_index_name}', num_sample))
                            get_num_sample -= num_sample
                            sample['num_sample'] = 0

    return gpu2city_map_index


def resample_by_geo(batch_size, infos, data_city_names, dataset, gpu2city_map_index, token2map_index):
    assert len(infos) == len(data_city_names) == len(token2map_index)
    num_geo = batch_size
    gpu2sample_id = {i: list() for i in range(num_geo)}
    gpu2timestamp = {i: list() for i in range(num_geo)}
    for i, (info, data_city_name) in enumerate(zip(infos, data_city_names)):
        i_gpu = False
        cur_map_index = token2map_index[info['token']]['map_index_list'][0]
        for gpu_id, city_map_index_list in gpu2city_map_index.items():
            for j, (city_name, city_pop_sample) in enumerate(city_map_index_list):
                city_bool = (city_name.split('_')[0] == data_city_name)
                map_index_bool = (int(city_name.split('_')[-2]) == cur_map_index[0]) and \
                                 (int(city_name.split('_')[-1]) == cur_map_index[1])
                if city_bool and map_index_bool and (city_pop_sample > 0):
                    city_pop_sample -= 1
                    city_map_index_list[j] = (city_name, city_pop_sample)
                    gpu2sample_id[gpu_id].append(i)
                    gpu2timestamp[gpu_id].append(info['timestamp'])
                    i_gpu = True
                if i_gpu:
                    break
            if i_gpu:
                break

    for i in range(num_geo):
        per_gpu_city = []
        for sid in gpu2sample_id[i]:
            map_index = token2map_index[infos[sid]['token']]['map_index_list'][0]
            per_gpu_city.append('_'.join([data_city_names[sid], 'map', str(map_index[0]), str(map_index[1])]))
        per_gpu_city = set(per_gpu_city)
        assert per_gpu_city == set([city_name for city_name, city_pop_sample in gpu2city_map_index[i]]), (
            per_gpu_city, set([city_name for city_name, city_pop_sample in gpu2city_map_index[i]]))

    for k, v in gpu2sample_id.items():
        # if dataset == 'val':
        # if dataset == 'val' or dataset == 'train':
        if dataset == 'train':
            print('sort by timestamp')
            timestamps = gpu2timestamp[k]
            ind = np.argsort(timestamps)
            gpu2sample_id[k] = np.array(v)[ind].tolist()
        print(k, len(v))
    return gpu2sample_id


def creat_map_gpu_by_name(map_slice_dict, map_attribute,
                          min_geo_loc, max_geo_loc,
                          gpu_city_list=None):
    gm_tile_size = map_attribute['global_map_tile_size']
    gm_raster_size = map_attribute['global_map_raster_size']

    for city_map_index_name in tqdm(gpu_city_list):
        city_name = city_map_index_name.split('_')[0]
        map_index = [int(i) for i in city_map_index_name.split('_')[-2:]]
        city_min_bound = min_geo_loc[city_name]
        city_max_bound = max_geo_loc[city_name]

        tile_interval = map_tile_interval(city_min_bound, city_max_bound, gm_tile_size)
        tile_grid_size = map_tile_raster_size(tile_interval, gm_raster_size)

        i, j = map_index[0], map_index[1]
        if f'map_{city_name}_{i}_{j}' not in map_slice_dict:
            map_slice_dict[f'map_{city_name}_{i}_{j}'] = \
                creat_empty_map_tile_tensor(tile_grid_size, **map_attribute['tile_param'])
        else:
            map_slice_dict[f'map_{city_name}_{i}_{j}'].zero_()


if __name__ == '__main__':
    map_attribute = {
        'root_dir': '/localdata_ssd/map_slices/raster_global_map',
        'prefix': 'lane_render_predict_video_2',
        'type': 'rasterized',
        'batch_size': 8,
        'tile_param': {
            'data_type': 'float32',
            'embed_dims': 3,
            'num_traversals': 1, },
        'global_map_tile_size': [1, 1],
        'global_map_raster_size': [0.15, 0.15],
    }
    bev_attribute = {
        'real_h': 30,
        'real_w': 60,
        'bev_h': 200,
        'bev_w': 400
    }
    dataset = 'val'
    result_root = '/home/xiongx/repository/marsmap/vis_results/bevformer_centeraware'
    # result_root = '/home/xiongx/repository/marsmap/vis_results/GT'

    nusc_min_geo_loc, nusc_max_geo_loc = map_geo_loc(dataset)
    print(map_attribute, dataset, nusc_min_geo_loc, nusc_max_geo_loc)

    map_tile_dict = {}
    creat_global_map(map_tile_dict, map_attribute, nusc_min_geo_loc, nusc_max_geo_loc)

    nusc_data_infos_val = load_nusc_data_infos(dataset)
    nusc_data_city_names_val = load_nusc_data_cities(dataset, root=None)

    val_token2map_index, val_map_index2token = get_token2map_index(
        nusc_data_infos_val, nusc_data_city_names_val,
        nusc_min_geo_loc, nusc_max_geo_loc,
        map_attribute['global_map_tile_size'])

    nusc_data_infos_val_timestamp = list(sorted(nusc_data_infos_val, key=lambda e: e['timestamp']))
    dataset_cities = [key for key in nusc_min_geo_loc]
    map_index2count = init_city_map_tile_dict(dataset_cities, map_attribute['global_map_tile_size'])
    for num_info, info in tqdm(enumerate(nusc_data_infos_val_timestamp)):
        # for num_info, info in tqdm(enumerate(nusc_data_infos_val)):
        data_cn = val_token2map_index[info['token']]['loc']
        if data_cn != 'singapore-hollandvillage':
            continue
        map_index = val_token2map_index[info['token']]['map_index_list'][0]

        map_slice_min_bound, map_slice_max_bound = map_tile_bound(
            nusc_min_geo_loc, nusc_max_geo_loc, data_cn, map_index,
            map_attribute['global_map_tile_size'])

        trans = gen_matrix(
            info['ego2global_rotation'],
            info['ego2global_translation']
        )

        global_map_slice = map_tile_dict[f'map_{data_cn}_{map_index[0]}_{map_index[1]}']
        try:
            bev_feature = torch.load(os.path.join(result_root, info['token'])).astype(np.int32)
            bev_feature_org = copy.deepcopy(bev_feature)
        except:
            print('im here')
            continue
        bev_feature = bev_feature.transpose(1, 2, 0)[np.newaxis]

        if not isinstance(global_map_slice, np.ndarray):
            global_map_slice = global_map_slice.detach()

        get_coords_resample(
            bev_feature,
            None,
            global_map_slice,
            trans,
            global_bound_h=(map_slice_min_bound[1],
                            map_slice_max_bound[1]),
            global_bound_w=(map_slice_min_bound[0],
                            map_slice_max_bound[0]),
            real_h=bev_attribute['real_h'],
            real_w=bev_attribute['real_w'],
            bev_h=bev_attribute['bev_h'],
            bev_w=bev_attribute['bev_w'],
            flag='replace'
        )
        map_tile_dict[f'map_{data_cn}_{map_index[0]}_{map_index[1]}'] = global_map_slice

        map_index2count[data_cn][f'map_index_{map_index[0]}_{map_index[1]}'].append(1)

        result_root_gt = '/home/xiongx/repository/marsmap/vis_results/GT'
        bev_feature_gt = torch.load(os.path.join(result_root_gt, info['token'])).astype(np.int32)
        result_root_baseline = '/home/xiongx/repository/marsmap/vis_results/bevformer_baseline'
        bev_feature_baseline = torch.load(os.path.join(result_root_baseline, info['token'])).astype(np.int32)

        result_root_hdmapnet = '/home/xiongx/repository/marsmap/vis_results/hdmapnet_baseline'
        bev_feature_hdmapnet = torch.load(os.path.join(result_root_hdmapnet, info['token'])).astype(np.int32)

        fig = plt.figure(figsize=(4, 8), dpi=200)
        row_num = 4
        for i, semantic_pro in enumerate([bev_feature_hdmapnet, bev_feature_baseline, bev_feature_org, bev_feature_gt]):
            ax = fig.add_subplot(row_num, 1, i + 1)
            semantic_show = np.full(
                shape=(3, semantic_pro.shape[-2],
                       semantic_pro.shape[-1]),
                fill_value=255, dtype=np.uint8)

            semantic_show[:, semantic_pro[2].astype(np.int32) == 1] = \
                np.array([107, 156, 123], dtype=np.uint8).reshape(3, 1)
            if i == 5:
                semantic_show[:, semantic_pro[0].astype(np.int32) == 1] = \
                    np.array([178, 97, 103], dtype=np.uint8).reshape(3, 1)
                semantic_show[:, semantic_pro[1].astype(np.int32) == 1] = \
                    np.array([97, 135, 178], dtype=np.uint8).reshape(3, 1)
            else:
                semantic_show[:, semantic_pro[0].astype(np.int32) == 1] = \
                    np.array([97, 135, 178], dtype=np.uint8).reshape(3, 1)
                semantic_show[:, semantic_pro[1].astype(np.int32) == 1] = \
                    np.array([178, 97, 103], dtype=np.uint8).reshape(3, 1)

            ax.imshow(semantic_show.transpose((1, 2, 0)))
            ax.set_xbound(0, 400)
            ax.set_ybound(0, 200)
            ax.axis('off')

        save_vis_pred_dir = os.path.join('/localdata_ssd/map_slices', map_attribute['prefix'], data_cn, 'zoomin')
        os.makedirs(save_vis_pred_dir, exist_ok=True)

        count = len(map_index2count[data_cn][f'map_index_{map_index[0]}_{map_index[1]}'])
        count = str(count).zfill(4)
        imname = f'{save_vis_pred_dir}/map_{count}.jpg'
        print('saving', imname)
        plt.savefig(imname)
        plt.close()

        # bound = 6000
        # pointer = 0
        # print(bound * pointer)
        # print(bound * (pointer + 1))
        # if bound * pointer < num_info < bound * (pointer + 1):
        #     semantic_show = np.full(
        #         shape=(3, global_map_slice.shape[1],
        #                global_map_slice.shape[2]),
        #         fill_value=255, dtype=np.uint8)
        #     semantic_show[:, global_map_slice[0, ..., 0].astype(np.int32) == 1] = \
        #         np.array([97, 135, 178], dtype=np.uint8).reshape(3, 1)
        #     semantic_show[:, global_map_slice[0, ..., 1].astype(np.int32) == 1] = \
        #         np.array([178, 97, 103], dtype=np.uint8).reshape(3, 1)
        #     semantic_show[:, global_map_slice[0, ..., 2].astype(np.int32) == 1] = \
        #         np.array([107, 156, 123], dtype=np.uint8).reshape(3, 1)
        #
        #     fig = plt.figure(figsize=(7, 7), dpi=400)
        #     ax = fig.add_subplot(1, 1, 1)
        #     ax.imshow(semantic_show.transpose((1, 2, 0)))
        #     ax.set_xbound(0, global_map_slice.shape[2])
        #     ax.set_ybound(0, global_map_slice.shape[1])
        #     ax.axis('off')
        #
        #     save_vis_pred_dir = os.path.join('/localdata_ssd/map_slices', map_attribute['prefix'], data_cn)
        #     os.makedirs(save_vis_pred_dir, exist_ok=True)
        #
        #     count = len(map_index2count[data_cn][f'map_index_{map_index[0]}_{map_index[1]}'])
        #     imname = f'{save_vis_pred_dir}/map_{count}.jpg'
        #     print('saving', imname)
        #     plt.savefig(imname)
        #     plt.close()

    cus_map_path = map_path(map_attribute, dataset)
    save_global_map(cus_map_path, map_tile_dict, map_attribute, nusc_min_geo_loc, nusc_max_geo_loc)

    cus_map_path = map_path(map_attribute, dataset)
    map_tile_names = glob.glob(f'{cus_map_path}/*/**.pth')
    for map_tile_name in map_tile_names:
        map_tile = torch.load(map_tile_name)
        semantic_show = np.full(
            shape=(3, map_tile.shape[1],
                   map_tile.shape[2]),
            fill_value=255, dtype=np.uint8)
        semantic_show[:, map_tile[0, ..., 0].astype(np.int32) == 1] = \
            np.array([97, 135, 178], dtype=np.uint8).reshape(3, 1)
        semantic_show[:, map_tile[0, ..., 1].astype(np.int32) == 1] = \
            np.array([178, 97, 103], dtype=np.uint8).reshape(3, 1)
        semantic_show[:, map_tile[0, ..., 2].astype(np.int32) == 1] = \
            np.array([107, 156, 123], dtype=np.uint8).reshape(3, 1)

        fig = plt.figure(figsize=(10, 10), dpi=600)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(semantic_show.transpose((1, 2, 0)))
        ax.set_xbound(0, map_tile.shape[2])
        ax.set_ybound(0, map_tile.shape[1])
        ax.axis('off')

        save_vis_pred_dir = map_attribute['prefix']
        os.makedirs(save_vis_pred_dir, exist_ok=True)

        city_name = map_tile_name.split('/')[-2]
        map_index = map_tile_name.split('/')[-1][:-4]
        imname = f'{save_vis_pred_dir}/map_{city_name}_{map_index}.jpg'
        print('saving', imname)
        plt.savefig(imname)
        plt.close()

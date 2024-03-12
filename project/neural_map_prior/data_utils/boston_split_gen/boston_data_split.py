import pickle as pkl
from copy import deepcopy

import numpy as np

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


def load_token2traversal_id(dataset):
    with open(f'/home/xiongx/repository/marsmap/global_map_construction/{dataset}_token2traversal_id.pkl', 'rb') as f:
        token2traversal_id = pkl.load(f)
    return token2traversal_id


def load_data_infos(dataset, only_city=False):
    with open(f'/home/xiongx/repository/marsmap/global_map_construction/{dataset}_infos_cityname.pkl', 'rb') as f:
        data_city_names = pkl.load(f)
    if only_city:
        return data_city_names
    with open(f'/public/MARS/datasets/nuScenes/nuscenes_infos_temporal_{dataset}.pkl', 'rb') as f:
        # with open(f'/localdata_ssd/nuScenes/nuscenes_infos_{dataset}.pkl', 'rb') as f:
        infos = pkl.load(f)['infos']
    print(f'load {dataset} data infos for nuscenes dataset...')
    return infos, data_city_names


data_infos_train, data_city_names_train = load_data_infos('train')
data_infos_val, data_city_names_val = load_data_infos('val')
data_infos = data_infos_train + data_infos_val
data_city_names = data_city_names_train + data_city_names_val
print('data_infos length: ', len(data_infos))
print('data_city_names length: ', len(data_city_names))

max_trip = 0
print('max_trip: ', max_trip)

train_token2traversal_id = load_token2traversal_id('train')
val_token2traversal_id = load_token2traversal_id('val')
token2traversal_id = deepcopy(train_token2traversal_id)
token2traversal_id.update(val_token2traversal_id)
print('token2traversal_id length: ', len(token2traversal_id))

train_boston_data_infos = []
val_boston_data_infos = []
train_boston_data_cities = []
val_boston_data_cities = []
train_boston_token = []
val_boston_token = []

city_range = train_max_geo_loc['boston-seaport'] - train_min_geo_loc['boston-seaport']
print('city_range: ', city_range)
city_middle = train_min_geo_loc['boston-seaport'] + (city_range / 2)
print('city_middle: ', city_middle)

for info, city_name in zip(data_infos, data_city_names):
    if city_name == 'boston-seaport':
        if info['ego2global_translation'][1] < city_middle[1] and (
                info['token'] in train_token2traversal_id or (
                info['token'] in val_token2traversal_id)):
            if max_trip > 0:
                if token2traversal_id[info['token']] <= max_trip:
                    train_boston_data_infos.append(deepcopy(info))
                    train_boston_data_cities.append(deepcopy(city_name))
            else:
                train_boston_data_infos.append(deepcopy(info))
                train_boston_data_cities.append(deepcopy(city_name))
        if info['ego2global_translation'][1] >= city_middle[1] and (
                info['token'] in train_token2traversal_id or (
                info['token'] in val_token2traversal_id)):
            if max_trip > 0:
                if token2traversal_id[info['token']] <= max_trip:
                    val_boston_data_infos.append(deepcopy(info))
                    val_boston_data_cities.append(deepcopy(city_name))
            else:
                val_boston_data_infos.append(deepcopy(info))
                val_boston_data_cities.append(deepcopy(city_name))

print('train_boston_data_infos length: ', len(train_boston_data_infos))
print('val_boston_data_infos length: ', len(val_boston_data_infos))
print('train_boston_data_cities length: ', len(train_boston_data_cities))
print('val_boston_data_cities length: ', len(val_boston_data_cities))

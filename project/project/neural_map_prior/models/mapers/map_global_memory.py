from copy import deepcopy
import pickle as pkl

from project.neural_map_prior.map_tiles.lane_render import \
    load_nusc_data_infos, \
    load_nusc_data_cities, \
    split_nusc_geo_bs, \
    get_token2map_index, \
    map_geo_loc, \
    creat_map_gpu_by_name, \
    map_tile_bound, \
    get_coords_resample


class MapGlobalMemory(object):
    def __init__(self, map_attribute, bev_attribute):
        self.bev_attribute = bev_attribute
        self.map_attribute = map_attribute

        if 'nusc_city_split' in map_attribute and map_attribute['nusc_city_split']:
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
            train_infos = new_train_infos
            train_data_cities = new_train_data_cities

        elif 'nusc_new_split' in map_attribute and map_attribute['nusc_new_split']:
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
            train_infos = new_train_infos
            train_data_cities = new_train_data_cities

        else:
            train_infos, train_data_cities = self.gen_token2map_info('train')

        print('training new_data_infos', len(train_infos))
        print('training new_data_city', len(train_data_cities))
        if ('nusc_city_split' in map_attribute and map_attribute['nusc_city_split']) or (
                'nusc_new_split' in map_attribute and map_attribute['nusc_new_split']):
            dataset = 'city'
        else:
            dataset = 'train'
        self.nusc_train_min_geo_loc, self.nusc_train_max_geo_loc = \
            map_geo_loc(dataset)
        self.train_token2map_index, self.train_map_index2token = \
            get_token2map_index(
                train_infos,
                train_data_cities,
                self.nusc_train_min_geo_loc,
                self.nusc_train_max_geo_loc,
                map_attribute['global_map_tile_size'])
        self.train_gpu2city = split_nusc_geo_bs(
            self.map_attribute['batch_size'],
            train_infos,
            train_data_cities,
            dataset,
            self.train_token2map_index,
            self.map_attribute['global_map_tile_size'],
            self.train_map_index2token)

        if 'nusc_city_split' in map_attribute and map_attribute['nusc_city_split']:
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
            val_infos = new_val_infos
            val_data_cities = new_val_data_cities

        elif 'nusc_new_split' in map_attribute and map_attribute['nusc_new_split']:
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
            val_infos = new_val_infos
            val_data_cities = new_val_data_cities

        else:
            val_infos, val_data_cities = self.gen_token2map_info('val')

        print('training new_data_infos', len(val_infos))
        print('training new_data_city', len(val_data_cities))
        if ('nusc_city_split' in map_attribute and map_attribute['nusc_city_split']) or (
                'nusc_new_split' in map_attribute and map_attribute['nusc_new_split']):
            dataset = 'city'
        else:
            dataset = 'val'
        self.nusc_val_min_geo_loc, self.nusc_val_max_geo_loc = \
            map_geo_loc(dataset)
        self.val_token2map_index, self.val_map_index2token = \
            get_token2map_index(
                val_infos,
                val_data_cities,
                self.nusc_val_min_geo_loc,
                self.nusc_val_max_geo_loc,
                self.map_attribute['global_map_tile_size'])
        self.val_gpu2city = split_nusc_geo_bs(
            self.map_attribute['batch_size'],
            val_infos,
            val_data_cities,
            dataset,
            self.val_token2map_index,
            self.map_attribute['global_map_tile_size'],
            self.val_map_index2token)

        self.map_slice_float_dict = {}
        self.map_slice_int_dict = {}
        self.map_center_dis_dict = {}
        self.map_slice_onehot_dict = {}

        self.train_epoch_point = -2
        self.val_epoch_point = -2

    def gen_token2map_info(self, split):
        data_infos = load_nusc_data_infos(split)
        data_cities = load_nusc_data_cities(split)

        if not (self.map_attribute['batch_size'] == 1):
            tail = len(data_infos) % self.map_attribute['batch_size']
            data_infos = data_infos[:-tail]
            data_cities = data_cities[:-tail]
            assert len(data_infos) == len(data_cities), \
                (len(data_infos), len(data_cities))

        return data_infos, data_cities

    @staticmethod
    def gen_map_slice_int(map_attribute):
        map_attribute = deepcopy(map_attribute)
        tile_param = map_attribute['tile_param']
        tile_param['data_type'] = 'int16'
        tile_param['embed_dims'] = 1
        return map_attribute

    def reset_define_map(self, epoch, gpu_id, dataset, map_slices_name, map_attribute_func=None):
        map_attribute = deepcopy(self.map_attribute)
        if self.check_epoch(epoch, dataset):

            if isinstance(map_slices_name, str):
                map_slice_dict = getattr(self, map_slices_name)
                if map_attribute_func is not None:
                    map_attribute = map_attribute_func[map_slices_name](map_attribute)
                self.reset_map(epoch, gpu_id, dataset, map_slice_dict, map_attribute)
                print(f'create empty map for: {map_slices_name}, {map_attribute}')

            elif isinstance(map_slices_name, list):
                for map_name in map_slices_name:
                    map_slice_dict = getattr(self, map_name)
                    if map_attribute_func is not None:
                        if map_name in map_attribute_func:
                            map_attribute = map_attribute_func[map_name](map_attribute)
                    self.reset_map(epoch, gpu_id, dataset, map_slice_dict, map_attribute)
                    print(f'create empty map for: {map_name}, {map_attribute}')

    def check_epoch(self, epoch, dataset):
        if ((self.train_epoch_point != epoch) and (dataset == 'train')) or \
                ((self.val_epoch_point != epoch) and (dataset == 'val')):

            if dataset == 'train':
                print('epoch:', epoch, 'self.train_epoch_point:', self.train_epoch_point)
                self.train_epoch_point = epoch
                return True
            elif dataset == 'val':
                print('epoch:', epoch, 'self.val_epoch_point:', self.val_epoch_point)
                self.val_epoch_point = epoch
                return True

    def reset_map(self, epoch, gpu_id, dataset, map_slices_dict, map_attribute):
        for k in map_slices_dict.keys():
            map_slices_dict[k].zero_()

        gpu_city_list = [city_name for (city_name, city_pop_sample) in getattr(self, f'{dataset}_gpu2city')[gpu_id]]
        min_geo_loc = getattr(self, f'nusc_{dataset}_min_geo_loc')
        max_geo_loc = getattr(self, f'nusc_{dataset}_max_geo_loc')

        creat_map_gpu_by_name(
            map_slices_dict,
            map_attribute,
            min_geo_loc, max_geo_loc,
            gpu_city_list=gpu_city_list)

        count = 0
        for k in map_slices_dict.keys():
            count += map_slices_dict[k].sum()
        assert count <= 1e-3
        print('gpu_id:', gpu_id, 'count:', count, 'gpu_city_list', gpu_city_list)

    def gen_map_info(self, token, split):
        city_name = getattr(self, f'{split}_token2map_index')[token]['loc']
        map_index = getattr(self, f'{split}_token2map_index')[token]['map_index_list'][0]
        map_slice_min_bound, map_slice_max_bound = map_tile_bound(
            getattr(self, f'nusc_{split}_min_geo_loc'),
            getattr(self, f'nusc_{split}_max_geo_loc'),
            city_name,
            map_index,
            self.map_attribute['global_map_tile_size'])
        return city_name, map_index, map_slice_min_bound, map_slice_max_bound

    def take_map_prior(self, bev_feature, token, img_meta, dataset, trans):
        city_name, map_index, map_slice_min_bound, map_slice_max_bound = self.gen_map_info(token, dataset)
        global_map_slice = self.map_slice_float_dict[f'map_{city_name}_{map_index[0]}_{map_index[1]}']
        get_coords_resample(
            bev_feature,
            None,
            global_map_slice.detach(),
            trans,
            global_bound_h=(
                map_slice_min_bound[1],
                map_slice_max_bound[1]),
            global_bound_w=(
                map_slice_min_bound[0],
                map_slice_max_bound[0]),
            real_h=self.bev_attribute['real_h'],
            real_w=self.bev_attribute['real_w'],
            bev_h=self.bev_attribute['bev_h'],
            bev_w=self.bev_attribute['bev_w'],
            reverse=True,
            type_flag='torch'
        )

    def replace_map_prior(self, bev_feature, token, img_metas, dataset, trans):
        city_name, map_index, map_slice_min_bound, map_slice_max_bound = self.gen_map_info(token, dataset)
        global_map_slice = self.map_slice_float_dict[f'map_{city_name}_{map_index[0]}_{map_index[1]}']
        get_coords_resample(
            bev_feature.detach(),
            None,
            global_map_slice.detach(),
            trans,
            global_bound_h=(
                map_slice_min_bound[1],
                map_slice_max_bound[1]),
            global_bound_w=(
                map_slice_min_bound[0],
                map_slice_max_bound[0]),
            real_h=self.bev_attribute['real_h'],
            real_w=self.bev_attribute['real_w'],
            bev_h=self.bev_attribute['bev_h'],
            bev_w=self.bev_attribute['bev_w'],
            flag='replace',
            type_flag='torch'
        )

import numpy as np
import warnings

import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
from shapely import ops
from shapely.geometry import LineString, box, MultiPolygon, Polygon

from ..nuscences_utils.map_api import CNuScenesMapExplorer

warnings.filterwarnings("ignore")


@PIPELINES.register_module(force=True)
class VectorizeLocalMap(object):

    def __init__(self,
                 data_root="/mnt/datasets/nuScenes/",
                 patch_size=(30, 60),
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=10,
                 num_samples=250,
                 padding=True,
                 max_len=30,
                 normalize=True,
                 fixed_num=50,
                 sample_pts=True,
                 rep_vectors=False,
                 rep_vector_len=8,
                 class2label={
                     'ped_crossing': 0,
                     'divider': 1,
                     'contours': 2,
                     'others': -1,
                 }, **kwargs):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = data_root
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.contour_classes = contour_classes
        self.class2label = class2label
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = CNuScenesMapExplorer(self.nusc_maps[loc])

        self.layer2class = {
            'ped_crossing': 'ped_crossing',
            'lane_divider': 'divider',
            'road_divider': 'divider',
            'road_segment': 'contours',
            'lane': 'contours',
        }

        self.process_func = {
            'ped_crossing': self.ped_geoms_to_vectors,
            'divider': self.line_geoms_to_vectors,
            'contours': self.poly_geoms_to_vectors,
        }

        self.colors = {
            'ped_crossing': 'blue',
            'divider': 'orange',
            'contours': 'green',
            # origin type
            'lane_divider': 'orange',
            'road_divider': 'orange',
            'road_segment': 'green',
            'lane': 'green',
        }

        self.sample_pts = sample_pts

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.max_len = max_len
        self.normalize = normalize
        self.fixed_num = fixed_num

        self.rep_vectors = rep_vectors
        self.rep_vector_len = rep_vector_len

        self.size = np.array([self.patch_size[1], self.patch_size[0]]) + 2

    def retrive_geom(self, patch_params):
        '''
            Get the geometric data.
            Returns: dict
        '''
        patch_box, patch_angle, location = patch_params
        geoms_dict = {}
        layers = self.line_classes + self.ped_crossing_classes + self.contour_classes
        for layer_name in layers:

            # retrive the geo
            if layer_name in self.nusc_maps[location].non_geometric_line_layers:
                geoms = self.map_explorer[location]._get_layer_line(
                    patch_box, patch_angle, layer_name)
            elif layer_name in self.nusc_maps[location].non_geometric_polygon_layers:
                geoms = self.map_explorer[location]._get_layer_polygon(
                    patch_box, patch_angle, layer_name)
            else:
                raise ValueError('{} is not a valid layer'.format(layer_name))

            if geoms is None:
                continue

            # change every geoms set to list
            if not isinstance(geoms, list):
                geoms = [geoms, ]

            geoms_dict[layer_name] = geoms

        return geoms_dict

    def union_geoms(self, geoms_dict):

        customized_geoms_dict = {}

        # contour
        roads = geoms_dict['road_segment']
        lanes = geoms_dict['lane']
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])

        customized_geoms_dict['contours'] = ('contours', [union_segments, ])

        # ped
        union_ped = ops.unary_union(geoms_dict['ped_crossing'])
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])
        geoms_dict['ped_crossing'] = [union_ped, ]

        for layer_name, custom_class in self.layer2class.items():

            if custom_class == 'contours':
                continue

            customized_geoms_dict[layer_name] = (
                custom_class, geoms_dict[layer_name])

        return customized_geoms_dict

    def convert2vec(self, geoms_dict: dict, sample_pts=False, override_veclen: int = None):

        vector_dict = {}
        for layer_name, (customized_class, geoms) in geoms_dict.items():

            line_strings = self.process_func[customized_class](geoms)

            vector_len = self.fixed_num[customized_class]
            if override_veclen is not None:
                vector_len = override_veclen

            vectors = self._geom_to_vectors(
                line_strings, customized_class, vector_len, sample_pts)
            vector_dict.update({layer_name: (customized_class, vectors)})

        return vector_dict

    def _geom_to_vectors(self, line_geom, label, vector_len, sample_pts=False):
        '''
            transfrom the geo type 2 line vectors
        '''
        line_vectors = {'vectors': [], 'length': []}
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for l in line:
                        if sample_pts:
                            v, nl = self._sample_pts_from_line(l, label, vector_len)
                        else:
                            v, nl = self._geoms2pts(l, label, vector_len)
                        line_vectors['vectors'].append(v.astype(np.float))
                        line_vectors['length'].append(nl)
                elif line.geom_type == 'LineString':
                    if sample_pts:
                        v, nl = self._sample_pts_from_line(line, label, vector_len)
                    else:
                        v, nl = self._geoms2pts(line, label, vector_len)
                    line_vectors['vectors'].append(v.astype(np.float))
                    line_vectors['length'].append(nl)
                else:
                    raise NotImplementedError

        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geoms: list):

        results = []
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []

        for geom in polygon_geoms:
            for poly in geom:
                exteriors.append(poly.exterior)
                for inter in poly.interiors:
                    interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            # since the start and end will disjoint
            # after applying the intersection.
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if lines.type != 'LineString':
                lines = ops.linemerge(lines)
            results.append(lines)

        return results

    def ped_geoms_to_vectors(self, geoms: list):

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for geom in geoms:
            for ped_poly in geom:
                # rect = ped_poly.minimum_rotated_rectangle
                ext = ped_poly.exterior
                if not ext.is_ccw:
                    ext.coords = list(ext.coords)[::-1]
                lines = ext.intersection(local_patch)

                if lines.type != 'LineString':
                    lines = ops.linemerge(lines)

                # same instance but not connected.
                if lines.type != 'LineString':
                    ls = []
                    for l in lines.geoms:
                        ls.append(np.array(l.coords))

                    lines = np.concatenate(ls, axis=0)
                    lines = LineString(lines)

                results.append(lines)

        return results

    def line_geoms_to_vectors(self, geom):
        # XXX
        return geom

    def _geoms2pts(self, line, label, fixed_point_num):

        # if we still use the fix point
        if fixed_point_num > 0:
            remain_points = fixed_point_num - np.asarray(line.coords).shape[0]
            if remain_points < 0:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > fixed_point_num:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

                remain_points = fixed_point_num - \
                                np.asarray(line.coords).shape[0]
                if remain_points > 0:
                    line = self.pad_line_with_interpolated_line(
                        line, remain_points)

            elif remain_points > 0:

                line = self.pad_line_with_interpolated_line(
                    line, remain_points)

            v = line
            if not isinstance(v, np.ndarray):
                v = np.asarray(line.coords)

            valid_len = v.shape[0]

        elif self.padding:  # dynamic points

            if self.max_len < np.asarray(line.coords).shape[0]:

                tolerance = 0.4
                while np.asarray(line.coords).shape[0] > self.max_len:
                    line = line.simplify(tolerance, preserve_topology=True)
                    tolerance += 0.2

            v = np.asarray(line.coords)
            valid_len = v.shape[0]

            pad_len = self.max_len - valid_len
            v = np.pad(v, ((0, pad_len), (0, 0)), 'constant')

        else:
            # dynamic points without padding
            line = line.simplify(0.2, preserve_topology=True)
            v = np.array(line.coords)
            valid_len = len(v)

        if self.normalize:
            v = self.normalize_line(v)

        return v, valid_len

    def pad_line_with_interpolated_line(self, line: LineString, remain_points):
        ''' pad variable line with the interploated points'''

        origin_line = line
        line_length = line.length
        v = np.array(origin_line.coords)
        line_size = v.shape[0]

        interval = np.linalg.norm(v[1:] - v[:-1], axis=-1).cumsum()
        edges = np.hstack((np.array([0]), interval)) / line_length

        # padding points
        interpolated_distances = np.linspace(
            0, 1, remain_points + 2)[1:-1]  # get rid of start and end
        sampled_points = np.array([list(origin_line.interpolate(distance, normalized=True).coords)
                                   for distance in interpolated_distances]).reshape(-1, 2)

        # merge two line
        insert_idx = np.searchsorted(edges, interpolated_distances) - 1

        last_idx = 0
        new_line = []
        inserted_pos = np.unique(insert_idx)

        for i, idx in enumerate(inserted_pos):
            new_line += [v[last_idx:idx + 1], sampled_points[insert_idx == idx]]
            last_idx = idx + 1
        # for the remain points
        if last_idx <= line_size - 1:
            new_line += [v[last_idx:], ]

        merged_line = np.concatenate(new_line, 0)

        return merged_line

    def _sample_pts_from_line(self, line, label, fixed_point_num):

        if fixed_point_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, 2)
        else:
            # fixed number of points, so distance is line.length / self.fixed_num

            distances = np.linspace(0, line.length, fixed_point_num)
            sampled_points = np.array([list(line.interpolate(distance).coords)
                                       for distance in distances]).reshape(-1, 2)

        num_valid = len(sampled_points)

        # padding
        if fixed_point_num < 0 and self.padding:

            # fixed distance sampling need padding!
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate(
                    [sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

        if self.normalize:
            sampled_points = self.normalize_line(sampled_points)

        return sampled_points, num_valid

    def normalize_line(self, line):
        '''
            prevent extrime pts such as 0 or 1. 
        '''

        origin = -np.array([self.patch_size[1] / 2, self.patch_size[0] / 2])
        # for better learning
        line = line - origin
        line = line / self.size

        return line

    def debug_vis(self, patch_params, vectors_dict=None, geoms_dict=None, origin=False, token=''):

        import matplotlib.patheffects as pe
        patch_box, patch_angle, loc = patch_params
        flag = 0
        anno_opts = dict(size=10, path_effects=[pe.withStroke(linewidth=1, foreground="black")])

        if geoms_dict is not None:
            from matplotlib.patches import Polygon as mPolygon
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            # for k, (cname, geoms) in geoms_dict.items():
            for k, data in geoms_dict.items():
                if isinstance(data, tuple):
                    (cname, geom_list) = data
                else:
                    cname, geom_list = k, data

                for geom in geom_list:
                    if isinstance(geom, Polygon):
                        ax.add_patch(mPolygon(np.asarray(geom.exterior.coords),
                                              fill=True, color=self.colors[cname], alpha=.5, label=cname))
                        try:

                            if hasattr(geom, 'interiors') and len(geom.interiors) > 0:
                                for interior in geom.interiors:
                                    ax.add_patch(mPolygon(np.asarray(interior),
                                                          fill=True, color=self.colors[cname], alpha=.2, label=cname))
                                flag = 1
                        except:
                            import ipdb
                            ipdb.set_trace()
                    elif isinstance(geom, LineString):
                        xy = np.asarray(geom.coords)
                        ax.plot(xy[:, 0], xy[:, 1],
                                color=self.colors[cname], alpha=.5, label=cname)
                    elif 'Multi' in geom.geom_type:
                        for geo in geom:
                            ax.add_patch(mPolygon(np.asarray(geo.exterior.coords),
                                                  fill=True, color=self.colors[cname], alpha=.5, label=cname))

                            if hasattr(geo, 'interiors') and len(geo.interiors) > 0:
                                for interior in geo.interiors:
                                    ax.add_patch(mPolygon(np.asarray(interior),
                                                          fill=True, color=self.colors[cname], alpha=.2, label=cname))
                                flag = 1

            # npatch_box = np.asarray((60,30))
            # elimnate repeating label
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.set_xlim(-30, 30)
            ax.set_ylim(-15, 15)
            ax.axis('equal')
            ax.legend(by_label.values(), by_label.keys())
            ax.plot(0, 0, 'o')
            if flag:
                interior_flag = 'interior_'
            else:
                interior_flag = ''
            fig.savefig('./map_transform/{}{}_geoms.png'.format(interior_flag,
                                                                token), dpi=200, bbox_inches='tight')
            plt.close()

        if vectors_dict is not None:
            from matplotlib.patches import Polygon as mPolygon
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            for k, (cname, vector_dict) in vectors_dict.items():
                for vec, lt in zip(vector_dict['vectors'], vector_dict['line_type']):
                    xy = np.asarray(vec)
                    ax.plot(xy[:, 0], xy[:, 1],
                            color=self.colors[cname], alpha=.5, label=cname)
                    ax.plot(xy[:, 0], xy[:, 1], 'o',
                            color=self.colors[cname], alpha=.5, label=cname)

                    cp = xy.mean(0)
                    ax.text(cp[0], cp[1], lt, **anno_opts)

            # npatch_box = np.asarray((60,30))
            # elimnate repeating label
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.set_xlim(-30, 30)
            ax.set_ylim(-15, 15)
            ax.axis('equal')
            ax.legend(by_label.values(), by_label.keys())
            ax.plot(0, 0, 'o')
            fig.savefig('./map_transform/{}_vectorization.png'.format(token),
                        dpi=200, bbox_inches='tight')
            plt.close()

        if origin:
            # bitmap = BitMap(self.data_root, loc, 'basemap')
            bitmap = None

            # from ego to local
            global_patch = CNuScenesMapExplorer.get_patch_coord(
                patch_box, patch_angle)

            # visualized_layer = list(self.layer2class.keys())
            visualized_layer = self.nusc_maps[loc].non_geometric_layers
            # visualized_layer = [
            #     i for i in visualized_layer if i == 'road_segment']

            fig, ax = self.map_explorer[loc].render_map_patch(
                global_patch, visualized_layer, figsize=(10, 10), bitmap=bitmap)
            fig.savefig('./map_transform/{}_origin.png'.format(token),
                        dpi=200, bbox_inches='tight')

            plt.close()

    def get_global_patch(self, input_dict: dict):
        # transform to global coordination
        location = input_dict['location']
        ego2global_translation = input_dict['ego2global_translation']
        ego2global_rotation = input_dict['ego2global_rotation']
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)
        patch_box = (map_pose[0], map_pose[1],
                     self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        patch_params = (patch_box, patch_angle, location)
        return patch_params

    def vectorization(self, input_dict: dict):

        patch_params = self.get_global_patch(input_dict)

        # Retrive geo
        geoms_dict = self.retrive_geom(patch_params)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, orgin=False)

        # Optional union the data and convert customized labels
        geoms_dict = self.union_geoms(geoms_dict)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, origin=False, token=input_dict['token'])

        # Convert Geo 2 vec
        vectors_dict = self.convert2vec(geoms_dict, self.sample_pts)
        # self.debug_vis(patch_params, vectors_dict=vectors_dict,
        #                origin=False, token=input_dict['token'])

        # format the outputs list
        vectors = []
        for k, (custom_class, v) in vectors_dict.items():

            label = self.class2label.get(custom_class, -1)
            # filter out -1
            if label == -1:
                continue

            for vec, l in zip(v['vectors'], v['length']):
                # fliter out two small case
                # _vec = vec * self.size
                # _vlen = np.linalg.norm(_vec[1:]-_vec[:-1]).sum(-1)
                # if _vlen < 1:
                #     continue
                vectors.append((vec, l, label))

        input_dict['vectors'] = vectors

        if self.rep_vectors:

            rep_vectors_dict = \
                self.convert2vec(
                    geoms_dict, sample_pts=True, override_veclen=self.rep_vector_len)

            rep_vectors = []
            for k, (custom_class, v) in rep_vectors_dict.items():

                label = self.class2label.get(custom_class, -1)
                # filter out -1
                if label == -1:
                    continue

                for vec, l in zip(v['vectors'], v['length']):
                    rep_vectors.append((vec, l, label))

            input_dict['rep_vectors'] = rep_vectors

        return input_dict

    def __call__(self, input_dict: dict):

        input_dict = self.vectorization(input_dict)

        return input_dict

    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):

        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        patch_params = (patch_box, patch_angle, location)

        # Retrive geo
        geoms_dict = self.retrive_geom(patch_params)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, orgin=False)

        # Optional union the data and convert customized labels
        geoms_dict = self.union_geoms(geoms_dict)
        # self.debug_vis(patch_params, geoms_dict=geoms_dict, origin=False, token=input_dict['token'])

        # Convert Geo 2 vec
        vectors_dict = self.convert2vec(geoms_dict, self.sample_pts)
        # self.debug_vis(patch_params, vectors_dict=vectors_dict,
        #                origin=False, token=input_dict['token'])

        # format the outputs list
        vectors = []
        for k, (custom_class, v) in vectors_dict.items():

            label = self.class2label.get(custom_class, -1)
            # filter out -1
            if label == -1:
                continue

            for vec, l in zip(v['vectors'], v['length']):
                # fliter out two small case
                # _vec = vec * self.size
                # _vlen = np.linalg.norm(_vec[1:]-_vec[:-1]).sum(-1)
                # if _vlen < 1:
                #     continue
                vectors.append({
                    'pts': vec,
                    'pts_num': l,
                    'type': label
                })

        return vectors


if __name__ == '__main__':

    def load_annotations(ann_file, load_interval=1):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::load_interval]
        print('meta data', data['metadata'])
        return data_infos, data['metadata']['version'].split('-')[-1]


    root = '/public/MARS/datasets/nuScenes'
    # root="/mnt/datasets/nuScenes",
    cfg = dict(
        data_root=root,
        patch_size=(30, 60),
        sample_dist=0.7,
        num_samples=150,
        padding=False,
        normalize=True,
        # fixed_num={
        #     'ped_crossing': 10,
        #     'divider': 6,
        #     'contours': 30,
        #     'others': -1,
        # },
        fixed_num={
            'ped_crossing': -1,
            'divider': -1,
            'contours': -1,
            'others': -1,
        },
        sample_pts=False,
        class2label={
            'ped_crossing': 0,
            'divider': 1,
            'contours': 2,
            'others': -1,
        },
        rep_vectors=True,
        rep_vector_len=7,
    )

    infos, version = load_annotations(
        root + '/nuScences_map_trainval_infos_train.pkl')
    info = infos[0]

    vec = VectorizeLocalMap(**cfg)

    angle_class = 36
    direction_pred = True
    head_dim = 128
    patch_size = (30, 60)
    xbound = [-30.0, 30.0, 0.15]
    ybound = [-15.0, 15.0, 0.15]
    zbound = [-10.0, 10.0, 20.0]
    dbound = [4.0, 45.0, 1.0]
    rasterized_cfg = dict(
        raw_dataset_cfg=dict(
            version=version,
            data_root=root,
        ),
        data_aug_conf={
            'line_width': 5,
            'direction_pred': direction_pred,
            'angle_class': angle_class
        },
        grid_conf={
            'xbound': xbound,
            'ybound': ybound,
            'zbound': zbound,
            'dbound': dbound,
        }
    )

    # test the algorithm
    vec(info)
    if True:
        from tqdm import tqdm

        print('total:', len(infos))
        np.random.seed(11)
        select_list = np.random.randint(0, 28100, (100,))
        print(select_list)
        for i in tqdm(select_list):
            vs = vec(infos[i])

    if False:
        from tqdm import tqdm

        vectors = []
        print('total:', len(infos))
        for info in tqdm(infos):
            vectors.append(vec(info))

        print('total_number:', len(vectors))

        import mmcv

        mmcv.dump(vectors, 'vectors_key_points.pkl')

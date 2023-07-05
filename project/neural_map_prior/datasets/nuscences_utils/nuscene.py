from typing import Tuple, Optional, List

import cv2
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer, Geometry
from pyquaternion import Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, box, MultiPolygon, Polygon

from .utils import get_discrete_degree


class MyNuScenesMap(NuScenesMap):
    def __init__(self,
                 dataroot: str = '/data/sets/nuscenes',
                 map_name: str = 'singapore-onenorth'):
        super(MyNuScenesMap, self).__init__(dataroot, map_name)
        self.explorer = MyNuScenesMapExplorer(self)

    def get_map_mask(self,
                     patch_box: Optional[Tuple[float, float, float, float]],
                     patch_angle: float,
                     layer_names: List[str] = None,
                     canvas_size: Optional[Tuple[int, int]] = (100, 100),
                     thickness=5,
                     type='index',
                     angle_class=36) -> np.ndarray:
        """
        Return list of map mask layers of the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, this plots the entire map.
        :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
        :param canvas_size: Size of the output mask (h, w). If None, we use the default resolution of 10px/m.
        :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
        """
        return self.explorer.get_map_mask(patch_box, patch_angle, layer_names=layer_names, canvas_size=canvas_size,
                                          thickness=thickness, type=type, angle_class=angle_class)


class MyNuScenesMapExplorer(NuScenesMapExplorer):
    def __init__(self,
                 map_api: MyNuScenesMap,
                 representative_layers: Tuple[str] = ('drivable_area', 'lane', 'walkway'),
                 color_map: dict = None):
        super(MyNuScenesMapExplorer, self).__init__(map_api, representative_layers, color_map)

    @staticmethod
    def mask_for_lines(lines, mask, id, thickness=5, type='index', angle_class=36):
        """
        Convert a Shapely LineString back to an image mask ndarray.
        :param lines: List of shapely LineStrings to be converted to a numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray line mask.
        """
        if lines.geom_type == 'MultiLineString':
            for line in lines:
                id += 1
                coords = np.asarray(list(line.coords), np.int32).reshape((-1, 2))
                if len(coords) < 2:
                    continue
                if type == 'backward':
                    coords = np.flip(coords, 0)
                if type == 'index':
                    cv2.polylines(mask, [coords], False, color=id, thickness=thickness)
                else:
                    for i in range(len(coords) - 1):
                        cv2.polylines(mask, [coords[i:]], False,
                                      color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class),
                                      thickness=thickness)
                    # cv2.polylines(mask, [coords], False, color=get_discrete_degree(coords[-1] - coords[-2]), thickness=thickness)
        else:
            id += 1
            coords = np.asarray(list(lines.coords), np.int32).reshape((-1, 2))
            if len(coords) < 2:
                return mask, id
            if type == 'backward':
                coords = np.flip(coords, 0)
            if type == 'index':
                cv2.polylines(mask, [coords], False, color=id, thickness=thickness)
            else:
                for i in range(len(coords) - 1):
                    cv2.polylines(mask, [coords[i:]], False,
                                  color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class),
                                  thickness=thickness)
                # cv2.polylines(mask, [coords], False, color=get_discrete_degree(coords[-1] - coords[-2]), thickness=thickness)

        return mask, id

    @staticmethod
    def mask_for_polygons(polygons, mask, idx):
        """
        Convert a polygon or multipolygon list to an image mask ndarray.
        :param polygons: List of Shapely polygons to be converted to numpy array.
        :param mask: Canvas where mask will be generated.
        :return: Numpy ndarray polygon mask.
        """
        if not polygons:
            return mask

        def int_coords(x):
            # function to round and convert to int
            return np.array(x).round().astype(np.int32)

        exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
        interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
        idx += 1
        cv2.fillPoly(mask, exteriors, idx)
        cv2.fillPoly(mask, interiors, 0)
        return mask, idx

    def _polygon_geom_to_mask(self,
                              layer_geom: List[Polygon],
                              local_box: Tuple[float, float, float, float],
                              layer_name: str,
                              canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert polygon inside patch to binary mask and return the map patch.
        :param layer_geom: list of polygons for each map layer
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param layer_name: name of map layer to be converted to binary map mask patch.
        :param canvas_size: Size of the output mask (h, w).
        :return: Binary map mask patch with the size canvas_size.
        """
        if layer_name not in self.map_api.non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x, patch_y, patch_h, patch_w = local_box

        patch = self.get_patch_coord(local_box)

        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]

        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(canvas_size, np.uint8)

        idx = 0
        for polygon in layer_geom:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                new_polygon = affinity.affine_transform(new_polygon,
                                                        [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_polygon = affinity.scale(new_polygon, xfact=scale_width, yfact=scale_height, origin=(0, 0))

                if new_polygon.geom_type == 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                map_mask, idx = self.mask_for_polygons(new_polygon, map_mask, idx)

        return map_mask, idx

    def _line_geom_to_mask(self,
                           layer_geom: List[LineString],
                           local_box: Tuple[float, float, float, float],
                           layer_name: str,
                           canvas_size: Tuple[int, int],
                           thickness,
                           type,
                           angle_class):
        """
        Convert line inside patch to binary mask and return the map patch.
        :param layer_geom: list of LineStrings for each map layer
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param layer_name: name of map layer to be converted to binary map mask patch.
        :param canvas_size: Size of the output mask (h, w).
        :return: Binary map mask patch in a canvas size.
        """
        patch_x, patch_y, patch_h, patch_w = local_box

        patch = self.get_patch_coord(local_box)

        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        map_mask = np.zeros(canvas_size)

        idx = 0
        for line in layer_geom:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
                new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))

                map_mask, idx = self.mask_for_lines(new_line, map_mask, idx, thickness=thickness, type=type,
                                                    angle_class=angle_class)
        return map_mask, idx

    def _layer_geom_to_mask(self,
                            layer_name: str,
                            layer_geom: List[Geometry],
                            local_box: Tuple[float, float, float, float],
                            canvas_size: Tuple[int, int],
                            thickness,
                            type,
                            angle_class):
        if layer_name in ['ped_crossing_line']:
            return self._line_geom_to_mask(layer_geom, local_box, layer_name, canvas_size, thickness, type, angle_class)
        elif layer_name in self.map_api.non_geometric_line_layers:
            return self._line_geom_to_mask(layer_geom, local_box, layer_name, canvas_size, thickness, type, angle_class)
        elif layer_name in self.map_api.non_geometric_polygon_layers:
            return self._polygon_geom_to_mask(layer_geom, local_box, layer_name, canvas_size)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def map_geom_to_mask(self,
                         map_geom: List[Tuple[str, List[Geometry]]],
                         local_box: Tuple[float, float, float, float],
                         canvas_size: Tuple[int, int],
                         thickness,
                         type,
                         angle_class):
        """
        Return list of map mask layers of the specified patch.
        :param map_geom: List of layer names and their corresponding geometries.
        :param local_box: The local patch box defined as (x_center, y_center, height, width), where typically
            x_center = y_center = 0.
        :param canvas_size: Size of the output mask (h, w).
        :return: Stacked numpy array of size [c x h x w] with c channels and the same height/width as the canvas.
        """
        # Get each layer mask and stack them into a numpy tensor.
        map_mask = []
        num_inst = []
        for layer_name, layer_geom in map_geom:
            layer_mask, layer_inst = self._layer_geom_to_mask(layer_name, layer_geom, local_box, canvas_size, thickness,
                                                              type, angle_class)
            if layer_mask is not None:
                map_mask.append(layer_mask)
                num_inst.append(layer_inst)

        return np.array(map_mask), np.array(num_inst)

    def _get_layer_geom(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_name: str) -> List[Geometry]:
        if layer_name in ['ped_crossing_line']:
            return self._get_ped_crossing_line(patch_box, patch_angle)
        elif layer_name in self.map_api.non_geometric_polygon_layers:
            return self._get_layer_polygon(patch_box, patch_angle, layer_name)
        elif layer_name in self.map_api.non_geometric_line_layers:
            return self._get_layer_line(patch_box, patch_angle, layer_name)
        else:
            raise ValueError("{} is not a valid layer".format(layer_name))

    def _get_ped_crossing_line(self, patch_box, patch_angle):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.map_api, 'ped_crossing')
        for record in records:
            polygon = self.extract_polygon(record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def get_map_mask(self,
                     patch_box: Optional[Tuple[float, float, float, float]],
                     patch_angle: float,
                     layer_names: List[str] = None,
                     canvas_size: Tuple[int, int] = (100, 100),
                     thickness=5,
                     type='index',
                     angle_class=36):
        """
        Return list of map mask layers of the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, this plots the entire map.
        :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
        :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
        :param canvas_size: Size of the output mask (h, w). If None, we use the default resolution of 10px/m.
        :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
        """
        # For some combination of parameters, we need to know the size of the current map.
        if self.map_api.map_name == 'singapore-onenorth':
            map_dims = [1585.6, 2025.0]
        elif self.map_api.map_name == 'singapore-hollandvillage':
            map_dims = [2808.3, 2922.9]
        elif self.map_api.map_name == 'singapore-queenstown':
            map_dims = [3228.6, 3687.1]
        elif self.map_api.map_name == 'boston-seaport':
            map_dims = [2979.5, 2118.1]
        else:
            raise Exception('Error: Invalid map!')

        # If None, return the entire map.
        if patch_box is None:
            patch_box = [map_dims[0] / 2, map_dims[1] / 2, map_dims[1], map_dims[0]]

        # If None, return all geometric layers.
        if layer_names is None:
            layer_names = self.map_api.non_geometric_layers

        # If None, return the specified patch in the original scale of 10px/m.
        if canvas_size is None:
            map_scale = 10
            canvas_size = np.array((patch_box[2], patch_box[3])) * map_scale
            canvas_size = tuple(np.round(canvas_size).astype(np.int32))

        # Get geometry of each layer.
        map_geom = self.get_map_geom(patch_box, patch_angle, layer_names)

        # Convert geometry of each layer into mask and stack them into a numpy tensor.
        # Convert the patch box from global coordinates to local coordinates by setting the center to (0, 0).
        local_box = (0.0, 0.0, patch_box[2], patch_box[3])
        map_mask, num_inst = self.map_geom_to_mask(map_geom, local_box, canvas_size, thickness=thickness, type=type,
                                                   angle_class=angle_class)
        assert np.all(map_mask.shape[1:] == canvas_size)

        return map_mask, num_inst


def gen_topdown_mask(nuscene, nusc_maps, sample_record, patch_size, canvas_size, seg_layers, thickness=5, type='index',
                     angle_class=36):
    sample_record_data = sample_record['data']
    sample_data_record = nuscene.get('sample_data', sample_record_data['LIDAR_TOP'])

    pose_record = nuscene.get('ego_pose', sample_data_record['ego_pose_token'])
    map_pose = pose_record['translation'][:2]
    rotation = Quaternion(pose_record['rotation'])

    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    scene_record = nuscene.get('scene', sample_record['scene_token'])
    log_record = nuscene.get('log', scene_record['log_token'])
    location = log_record['location']
    topdown_seg_mask, num_inst = nusc_maps[location].get_map_mask(patch_box, patch_angle, seg_layers, canvas_size,
                                                                  thickness=thickness, type=type,
                                                                  angle_class=angle_class)
    # topdown_seg_mask = np.flip(topdown_seg_mask, 1)  # left-right correction
    return topdown_seg_mask, num_inst


def extract_contour(topdown_seg_mask, canvas_size, thickness=5, type='index', angle_class=36):
    topdown_seg_mask[topdown_seg_mask != 0] = 255
    ret, thresh = cv2.threshold(topdown_seg_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(topdown_seg_mask.shape)
    patch = box(1, 1, canvas_size[1] - 2, canvas_size[0] - 2)
    idx = 0
    for cnt in contours:
        cnt = cnt.reshape((-1, 2))
        cnt = np.append(cnt, cnt[0].reshape(-1, 2), axis=0)
        line = LineString(cnt)
        line = line.intersection(patch)
        if isinstance(line, MultiLineString):
            line = ops.linemerge(line)
        line = line.simplify(tolerance=1.0)

        if isinstance(line, MultiLineString):
            for l in line:
                idx += 1
                coords = np.asarray(list(l.coords), np.int32).reshape((-1, 2))
                if len(coords) < 2:
                    continue
                if type == 'backward':
                    coords = np.flip(coords, 0)
                if type == 'index':
                    cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
                else:
                    for i in range(len(coords) - 1):
                        cv2.polylines(mask, [coords[i:]], False,
                                      color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class),
                                      thickness=thickness)
                    # cv2.polylines(mask, [coords], False, color=get_discrete_degree(coords[-1] - coords[-2]), thickness=thickness)
        elif isinstance(line, LineString):
            idx += 1
            coords = np.asarray(list(line.coords), np.int32).reshape((-1, 2))
            if len(coords) < 2:
                continue
            if type == 'backward':
                coords = np.flip(coords, 0)
            if type == 'index':
                cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
            else:
                for i in range(len(coords) - 1):
                    cv2.polylines(mask, [coords[i:]], False,
                                  color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class),
                                  thickness=thickness)
                # cv2.polylines(mask, [coords], False, color=get_discrete_degree(coords[-1] - coords[-2]), thickness=thickness)
    return mask, idx

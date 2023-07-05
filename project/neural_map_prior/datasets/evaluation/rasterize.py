import cv2
import numpy as np

from shapely import affinity
from shapely.geometry import LineString, box


class RasterizeVectors:

    def __init__(self, num_class, patch_size=(30, 60), canvas_size=(200, 400)):

        self.num_class = num_class

        self.patch_size = patch_size  # (y,x)
        self.canvas_size = canvas_size  # (y,x)
        self.patch_box = (0.0, 0.0, patch_size[0], patch_size[1])

        self.patch = self.get_patch_coord(self.patch_box)

    def get_patch_coord(self, patch_box, patch_angle=0.0):

        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, patch_angle, origin=(
            patch_x, patch_y), use_radians=False)

        return patch

    def mask_for_lines(self, lines: LineString, mask, thickness, color):
        ''' generate line mask '''

        coords = np.asarray(lines).astype(np.int32)
        coords = coords.reshape(-1, 2)

        cv2.polylines(mask, [coords], False, color=color, thickness=thickness)

        return mask

    def line_geom_to_mask(self, geom, map_mask, thickness, color):

        if geom.geom_type == 'MultiLineString':

            for single_line in geom:
                map_mask = self.mask_for_lines(
                    single_line, map_mask, thickness, color)
        else:

            map_mask = self.mask_for_lines(
                geom, map_mask, thickness, color)

        return map_mask

    def preprocess_data(self, vectors):

        patch_x, patch_y, patch_h, patch_w = self.patch_box

        canvas_h = self.canvas_size[0]
        canvas_w = self.canvas_size[1]

        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w

        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        # last one is the background
        vector_dict = dict([(i, []) for i in range(self.num_class + 1)])

        for vector in vectors:

            if vector['pts_num'] < 2:
                continue

            line = LineString(vector['pts'][:vector['pts_num']])

            line = line.intersection(self.patch)
            if line.is_empty:
                continue

            # Transform line to coordination with (0,0) origin.
            # from (-30,30) to (0,60)
            line = affinity.affine_transform(
                line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])

            # from patch to canvas range
            line = affinity.scale(
                line, xfact=scale_width, yfact=scale_height, origin=(0, 0))

            confidence_level = vector.get('confidence_level', 1)

            vector_dict[vector['type']].append(
                (line, confidence_level))

        return vector_dict

    def rasterize_map(self, vectors, thickness):

        vectors = self.preprocess_data(vectors)

        confidence_levels = [-1]
        map_masks = np.zeros((self.num_class + 1, *self.canvas_size), np.uint8)
        map_masks[-1] = np.ones_like(map_masks[-1])
        # generate map for each class
        for label in range(self.num_class):

            if len(vectors[label]) == 0:
                continue

            for idx, (vector, confidence_level) in enumerate(vectors[label]):
                map_masks[label] = self.line_geom_to_mask(
                    vector, map_masks[label], thickness, color=idx + 1)

                confidence_levels.append(confidence_level)

        return map_masks, confidence_levels

    def __call__(self, *args, **kwargs):

        return self.rasterize_map(*args, **kwargs)

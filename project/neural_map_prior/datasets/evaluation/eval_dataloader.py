import json

import mmcv
import numpy as np

from .rasterize import RasterizeVectors
from .vectorized_map import VectorizedLocalMap


class HDMapNetEvalDataset(object):
    def __init__(self,
                 dataroot: str,
                 ann_file: str,
                 result_path: str,
                 thickness,
                 max_line_count=100,
                 num_class=3,
                 xbound=[-30., 30., 0.15],
                 ybound=[-15., 15., 0.15],
                 class2label={
                     'ped_crossing': 0,
                     'road_divider': 1,
                     'lane_divider': 1,
                     'contours': 2,
                     'others': -1,
                 }):

        # load results
        if result_path.endswith('.pkl'):
            self.prediction = mmcv.load(result_path)
        else:
            with open(result_path, 'r') as f:
                self.prediction = json.load(f)

        predict_token = set(self.prediction['results'].keys())

        # load sample info
        self.load_interval = 1
        self.data_infos = self.load_annotations(ann_file)

        print('sample token number: ', len(self.data_infos))
        print('prediction token: ', len(predict_token))

        self.data_infos = [
            info for info in self.data_infos if info['token'] in predict_token]

        self.max_line_count = max_line_count
        self.num_class = num_class
        self.thickness = thickness

        # vectorization
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])  # y
        canvas_w = int(patch_w / xbound[2])  # x
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.vector_map = VectorizedLocalMap(
            dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size,
            class2label=class2label)

        # self.vector_map = VectorizeLocalMap(
        #     dataroot,
        #     patch_size=self.patch_size,
        #     canvas_size=self.canvas_size,
        #     sample_dist=0.7,
        #     num_samples=150,
        #     sample_pts=False,
        #     max_len=100,
        #     padding=False,
        #     normalize=False,
        #     fixed_num={
        #         'ped_crossing': -1,
        #         'divider': -1,
        #         'contours': -1,
        #         'others': -1,
        #     },
        #     class2label={
        #              'ped_crossing': 0,
        #              'divider': 1,
        #              'contours': 2,
        #              'others': -1,
        #          })

        self.rasterize_map = RasterizeVectors(
            self.num_class, self.patch_size, self.canvas_size)

        # hard code
        self.gt_vectors = {}

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        info = self.data_infos[idx]

        location = info['location']
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']

        if 'gt_map' in self.prediction:
            gt_map = self.prediction['gt_map'][info['token']]
        else:
            gt_vectors = self.vector_map.gen_vectorized_samples(
                location, ego2global_translation, ego2global_rotation)
            gt_map, _ = \
                self.rasterize_map(gt_vectors, self.thickness)

        if self.prediction['meta']['vector']:
            pred_vectors = self.prediction['results'][info['token']]['vectors']
            for i, p in enumerate(pred_vectors):
                pred_vectors[i]['pts'] = renormalized(p['pts'])
                # /(np.array((30,15)))
            pred_map, confidence_level = \
                self.rasterize_map(pred_vectors, self.thickness)
        else:
            pred_map = self.prediction['results'][info['token']]['map']
            confidence_level = self.prediction['results'][info['token']
            ]['confidence_level']

        confidence_level = np.array(
            confidence_level + [-1] * (self.max_line_count - len(confidence_level))).astype(np.float32)

        assert gt_map.dtype == np.uint8
        assert pred_map.dtype == np.uint8
        assert isinstance(idx, int)
        assert confidence_level.dtype == np.float32

        out = dict(pred_map=pred_map,
                   confidence_level=confidence_level,
                   gt_map=gt_map,
                   indexes=idx)

        self.gt_vectors[info['token']] = gt_vectors

        if 'aux_map' in self.prediction['results'][info['token']].keys():
            out['aux_map'] = self.prediction['results'][info['token']]['aux_map']

        return out


def renormalized(vector):
    vector = vector + 1

    return vector


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate nuScenes local HD Map Construction Results.')
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--eval_set', type=str, default='mini_val',
                        choices=['train', 'val', 'test', 'mini_train', 'mini_val'])

    args = parser.parse_args()

    dataset = HDMapNetEvalDataset(
        args.version, args.dataroot, args.eval_set, args.result_path, thickness=2)
    for i in range(dataset.__len__()):
        pred_vectors, confidence_level, gt_vectors = dataset.__getitem__(i)

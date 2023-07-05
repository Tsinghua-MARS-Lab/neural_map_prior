import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.pipelines.transforms import Resize

from mmdet3d.core.points import BasePoints


@PIPELINES.register_module()
class FormatBundleMap(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names=None, with_gt=False, with_label=True, process_img=True):
        super(FormatBundleMap, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label
        self.process_img = process_img

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if 'img' in results and self.process_img:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                if imgs.shape[0] > 6:
                    imgs = imgs.reshape(-1, 6, imgs.shape[-3], imgs.shape[-2], imgs.shape[-1])
                    results['img'] = DC(to_tensor(imgs[0]), stack=True)
                    results['img_hist'] = DC(to_tensor(imgs[1:]), stack=True)
                else:
                    results['img'] = DC(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)
        for key in [
            'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
            'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
            'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)

        if 'vectors' in results:
            vectors = results['vectors']
            # cause len(vectors) in different sample in a batch has different length,
            #   so we should stack=Flase

            # cause vectors[i] = Tuple(lines, length, label) has different shapes,
            # so, we have to set cpu_only=True
            results['vectors'] = DC(vectors, stack=False, cpu_only=True)

        # same with above
        if 'polys' in results:
            results['polys'] = DC(results['polys'], stack=False, cpu_only=True)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str


@PIPELINES.register_module()
class Normalize3D(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            results[key] = [mmcv.imnormalize(
                img, self.mean, self.std, self.to_rgb) for img in results[key]]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class Pad3D(object):
    """Pad the image & mask.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = [mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val) for img in results[key]]
            results[key] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['img_fixed_size'] = self.size
        results['img_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class ResizeMultiViewImages(Resize):
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        super().__init__(img_scale,
                         multiscale_mode,
                         ratio_range,
                         keep_ratio,
                         bbox_clip_border,
                         backend,
                         override)

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                tmp = [mmcv.imrescale(
                    image,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend) for image in results[key]]
                imgs = [t[0] for t in tmp]
                scale_factor = tmp[0][1]
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = imgs[0].shape[:2]
                h, w = results[key][0].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                tmp = [mmcv.imresize(
                    image,
                    (results['scale'][1], results['scale'][0]),  # size in mmcv.imresize is (w, h)
                    return_scale=True,
                    backend=self.backend) for image in results[key]]
                imgs = [t[0] for t in tmp]
                w_scale = tmp[0][1]
                h_scale = tmp[0][2]

            results[key] = imgs

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'] = [img.shape for img in imgs]
            # in case that there is no padding
            results['pad_shape'] = [img.shape for img in imgs]
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio


@PIPELINES.register_module()
class ResizeCameraImage(object):
    def __init__(self, fH, fW, H, W):
        self.scale = (fW / W, fH, H)
        self.finalsize = (fW, fH)

        self.post_rot = torch.eye(2)
        self.post_tran = torch.zeros(2)

    def __call__(self, results: dict):
        img = results['img']

        tmps, post_rots, post_trans = [], [], []
        for image in results['img']:
            (tmp, scaleW, scaleH) = mmcv.imresize(
                image,
                self.finalsize,
                return_scale=True,
                backend='cv2')
            tmps.append(tmp)

            rot_resize = torch.Tensor([[scaleW, 0],
                                       [0, scaleH]])
            post_rots.append(rot_resize @ self.post_rot)
            post_trans.append(rot_resize @ self.post_tran)

        results['img'] = tmps
        results.update({
            'post_trans': torch.stack(post_trans, dim=0),
            'post_rots': torch.stack(post_rots, dim=0),
        })

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


if __name__ == '__main__':
    import numpy as np

    results = {'img': torch.randn((900, 1600, 3))}
    func = ResizeCameraImage(fW=352, fH=128,
                             W=1600, H=900, )

    func(results)

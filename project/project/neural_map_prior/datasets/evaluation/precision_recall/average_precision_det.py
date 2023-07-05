# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from multiprocessing import Pool

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from .tgfg import tpfp_bbox, tpfp_det, tpfp_rbbox


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_test(det_bboxes,
              gt_bboxes,
              threshold=0.5):
    tp = det_bboxes[:, 0]
    fp = det_bboxes[:, 1]

    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes
    """
    cls_dets = [res[class_id] for res in det_results]

    cls_gts, cls_gts_mask = [], []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id

        cls_gts.append(ann['bboxes'][gt_inds])
        cls_gts_mask.append(ann['bbox_masks'][gt_inds])

    return cls_dets, cls_gts, cls_gts_mask


def _eval_map(det_results,
              annotations,
              threshold=0.5,
              num_classes=3,
              class_name=None,
              logger=None,
              tpfp_fn_name='vec',
              nproc=4):
    """Evaluate mAP of a dataset.

    Args:


        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )

        scale_ranges (list[tuple] | None): canvas_size
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
        tpfp_fn (callable | None): The function used to determine true/
            false positives. If None, :func:`tpfp_default` is used as default
            unless dataset is 'det' or 'vid' (:func:`tpfp_imagenet` in this
            case). If it is given as a function, then this function is used
            to evaluate tp & fp. Default None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_samples = len(det_results)
    num_classes = len(det_results[0])  # positive class num
    print('num_samples', num_samples)
    print('num_classes', num_classes)

    tpfp_fn_dict = {
        'vec': tpfp_det,
        'rxyxy': tpfp_rbbox,
        'convex': tpfp_bbox,
    }

    pool = Pool(nproc)
    eval_results = []
    for i, clsname in enumerate(class_name):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_mask = \
            get_cls_results(det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        # XXX
        # func_name = cls2func[clsname]
        tpfp_fn = tpfp_fn_dict[tpfp_fn_name]
        # tpfp_fn = tpfp_bbox

        # Trick for serialized
        # only top-level function can be serized
        # somehow use partitial the return function is defined
        # at the top level.
        tpfp_fn = partial(tpfp_fn, threshold=threshold)

        args = []
        # compute tp and fp for each image with multiple processes
        if tpfp_fn_name == 'vec':
            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_dets, cls_gts, *args))
        elif tpfp_fn_name in ['convex', 'rxyxy']:
            tpfp = pool.starmap(
                tpfp_fn,
                zip(cls_dets, cls_gts, cls_gts_mask, *args))
        # debug and testing
        # tpfp = tpfp_fn(cls_dets[0], cls_gts[0], cls_gts_mask[0],threshold=threshold)
        # tpfp = (tpfp,)

        # XXX
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = 0
        for j, bbox in enumerate(cls_gts):
            num_gts += bbox.shape[0]

        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[sort_inds]
        fp = np.hstack(fp)[sort_inds]

        # calculate recall and precision with tp and fp
        # num_det*num_res
        tp = np.cumsum(tp, axis=0)
        fp = np.cumsum(fp, axis=0)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        # calculate AP
        # if dataset != 'voc07' else '11points'
        mode = 'area'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    aps = []
    for cls_result in eval_results:
        if cls_result['num_gts'] > 0:
            aps.append(cls_result['ap'])
    mean_ap = np.array(aps).mean().item() if len(aps) else 0.0

    print_map_summary(
        mean_ap, eval_results, class_name=class_name, logger=logger)

    return mean_ap, eval_results


def eval_map(cfg: dict, logger=None):
    print('results path: {}'.format(cfg['result_path']))
    results_file = mmcv.load(cfg['result_path'])
    det_res = list(results_file['det_res']['results'].values())
    anns = list(results_file['det_res']['annotations'].values())

    if len(det_res) == 0:
        return 0, None

    if 'det_threshold' in cfg:
        threshold = cfg['det_threshold']
    else:
        threshold = 0.5

    mean_ap, eval_results = _eval_map(
        det_res,
        anns,
        threshold=threshold,
        num_classes=cfg['num_class'],
        class_name=cfg['class_name'],
        tpfp_fn_name=cfg['tpfp_fn_name'],
        logger=logger,
        nproc=16)

    return mean_ap, eval_results


def print_map_summary(mean_ap,
                      results,
                      class_name=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    label_names = class_name

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate nuScenes local HD Map Construction Results.')
    # parser.add_argument('--result_path', type=str,
    #                     default='./work_dirs/gen_aug_larger/results_nuscence.pkl')
    # parser.add_argument('--result_path', type=str,
    #                     default='/home/dcz/liuyicheng_temp/marsmap/work_dirs/gen_aug_larger/results_nuscence.pkl')
    parser.add_argument('--result_path', type=str,
                        default='/home/dcz/liuyicheng_temp/marsmap/work_dirs/Rpv_convex_full/results.pkl')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--class_name', type=list,
                        default=['ped_crossing', 'divider', 'contours'])
    parser.add_argument('--CD_threshold', type=int, default=10)

    args = parser.parse_args()

    args = vars(args)
    args['result_path'] = '/home/dcz/liuyicheng_temp/marsmap/work_dirs/Rpv_multiscale_full/multiscale.pkl'
    args['tpfp_fn_name'] = 'rxyxy'
    args['det_threshold'] = 0.5

    eval_map(args)
    # threshold = 0.5
    # pred_scores = np.array([0.7, 0.3, 0.5, 0.6, 0.55, 0.9, 0.4, 0.2, 0.4, 0.3])
    # pred_scores = pred_scores[np.argsort(-pred_scores)]

    # det_results_tp = np.array([1,0,1,0,0,1,0,0,1,1])
    # det_results_fp = np.array([0,1,0,1,1,0,0,0,0,0])
    # det_results = np.stack([det_results_tp, det_results_fp, pred_scores],axis=-1)

    # annotations = {
    #     'labels':np.array([0,0,0,0,0]),
    #     'bboxes':np.array([1,1,1,1,1]).reshape(-1,1)
    # }

    # _eval_map([[det_results,],],
    #           [annotations,],
    #           threshold=0.5,
    #           num_classes=1,
    #           class_name=['ped_crossing', 'divider', 'contours'],
    #           logger=None,
    #           tpfp_fn=None,
    #           nproc=1)

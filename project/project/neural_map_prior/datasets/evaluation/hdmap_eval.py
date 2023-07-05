from pprint import pprint

import numpy as np
import torch
import tqdm

from .chamfer_dist import ChamferDistance
from .eval_dataloader import HDMapNetEvalDataset
# Metrics
# from hdmap_devkit.evaluation.AP import instance_mask_AP
from .iou import get_batch_iou
from .utils import CaseLogger

# from hdmap_devkit.evaluation.chamfer_distance import semantic_mask_chamfer_dist_cum

SAMPLED_RECALLS = torch.linspace(0.1, 1, 10)
THRESHOLDS = [2, 4, 6]
label2class = ['ped', 'divider', 'boundary']


def get_val_info(cfg: dict = None, device=None, ):
    if device is None:
        device = torch.device('cuda:0')

    print('ann_file path: {}'.format(cfg['ann_file']))
    print('results path: {}'.format(cfg['result_path']))

    if 'class2label' not in cfg.keys():
        cfg['class2label'] = {
            'ped_crossing': 0,
            'road_divider': 1,
            'lane_divider': 1,
            'contours': 2,
            'others': -1,
        }

    dataset = HDMapNetEvalDataset(
        dataroot=cfg['dataroot'],
        ann_file=cfg['ann_file'],
        result_path=cfg['result_path'],
        num_class=cfg['num_class'],
        thickness=cfg['thickness'],
        xbound=cfg['xbound'],
        ybound=cfg['ybound'],
        class2label=cfg['class2label'])

    logger = CaseLogger('hdmap_detr', dataset.prediction,
                        dataset.data_infos, dataset.version)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['bsz'], shuffle=False, drop_last=False, num_workers=8)

    total_results = {}
    result_keys = ['CD1', 'CD2',
                   'CD_num1', 'CD_num2',
                   'intersect', 'union',
                   'aux_intersect', 'aux_union']

    for k in result_keys:
        total_results[k] = torch.zeros(cfg['num_class']).to(device)
    # AP_matrix = torch.zeros((cfg['num_class'], len(THRESHOLDS))).to(device)
    # AP_count_matrix = torch.zeros((cfg['num_class'], len(THRESHOLDS))).to(device)

    H = cfg['ybound'][1] - cfg['ybound'][0]
    W = cfg['xbound'][1] - cfg['xbound'][0]
    cd_dist = ChamferDistance(
        H, W, cfg['xbound'][2], cfg['ybound'][2], device, normalize=True)

    print('running eval...')
    for data in tqdm.tqdm(data_loader):

        for k, v in data.items():
            data[k] = v.to(device)

        pred_map = data['pred_map']
        if pred_map.shape[1] != cfg['num_class']:
            pred_map = pred_map[:, :-1]

        gt_map = data['gt_map'][:, :-1]
        # import ipdb; ipdb.set_trace()

        # IoU
        # bound_pred_map = out_bound_with_dis(pred_map)
        # bound_gt_map = out_bound_with_dis(gt_map)
        # print('bound_pred_map', bound_pred_map.shape)
        # print('bound_gt_map', bound_gt_map.shape)
        # intersect, union = get_batch_iou_bound(bound_pred_map, bound_gt_map, device)

        intersect, union = get_batch_iou(pred_map, gt_map, device)

        # criterion = get_batched_iou_no_reduction(data['pred_map'], data['gt_map'], device)
        # logger.save_packed_data(criterion, data['gt_map'], data['pred_map'], data['indexes'])
        # import ipdb
        # ipdb.set_trace()
        # if True:
        #     # token = data_loader.dataset.data_infos[data['indexes'][0]]['token']
        #     # vec = data_loader.dataset.prediction['results'][token]['vectors']
        #     # print(vec)
        #     info = data_loader.dataset.data_infos[data['indexes'][0]]
        #     location = info['location']
        #     ego2global_translation = info['ego2global_translation']
        #     ego2global_rotation = info['ego2global_rotation']
        #     gt_v = data_loader.dataset.vector_map.gen_vectorized_samples(
        #         location, ego2global_translation, ego2global_rotation)
        #     print(gt_v)
        # vis_map(data['pred_map'][0], 'pred')
        # vis_map(data['gt_map'][0], 'gt')
        # vis_map_contrastive(data['gt_map'][0], data['pred_map'][0])

        # Chamfer Distance
        CD1, CD2, CD_num1, CD_num2 = \
            cd_dist.semantic_mask_chamfer_dist_cum_parallel(pred_map, gt_map)

        # auxiliary IoU
        if 'aux_map' in data.keys():
            aux_intersect, aux_union = get_batch_iou(
                data['aux_map'], data['gt_map'], device)

        # instance_mask_AP(AP_matrix, AP_count_matrix, pred_map, gt_map, cfg['xbound'][2], cfg['ybound'][2],
        #                  confidence_level, THRESHOLDS, sampled_recalls=SAMPLED_RECALLS, device=device)

        batch_results = dict(
            intersect=intersect,
            union=union,
            CD1=CD1,
            CD2=CD2,
            CD_num1=CD_num1,
            CD_num2=CD_num2,
        )

        if 'aux_map' in data.keys():
            batch_results.update(dict(aux_intersect=aux_intersect,
                                      aux_union=aux_union, ))

        for k, v in batch_results.items():
            total_results[f'{k}'] += v

    CD_pred = total_results['CD1'] / total_results['CD_num1']
    CD_label = total_results['CD2'] / total_results['CD_num2']
    CD = (total_results['CD1'] + total_results['CD2']) / \
         (total_results['CD_num1'] + total_results['CD_num2'])

    _results_dict = {
        'iou': (total_results['intersect'] / total_results['union']).cpu().numpy(),
        'CD_pred': CD_pred.cpu().numpy(),
        'CD_label': CD_label.cpu().numpy(),
        'CD': CD.cpu().numpy(),
        # 'Average_precision': (AP_matrix / AP_count_matrix).numpy(),
    }

    if total_results['aux_intersect'].sum() != 0:
        _results_dict.update({'aux_iou': (
                total_results['aux_intersect'] / total_results['aux_union']).cpu().numpy()})
    # convert to tensorboard friendly format
    results_dict = {}
    for k, v in _results_dict.items():
        for i in range(cfg['num_class']):
            key = k + '_cls:{}'.format(label2class[i])
            if v[i].size > 1:
                for j in range(cfg['num_class']):
                    kk = key + '_threshold:{}'.format(THRESHOLDS[j])
                    results_dict[kk] = v[i, j]
            else:
                _v = v[i]
                if 'iou' in key:
                    _v = _v * 100
                results_dict[key] = np.around(_v, decimals=3)

    return results_dict


def out_bound_with_dis(map):
    area_ratio = 5
    part_level = 2
    print('area_ratio:', area_ratio)
    print('part_level:', part_level)
    B, C, H, W = map.shape
    print('hehehe:,', B, C, H, W)
    part_h = (H // area_ratio) * (part_level + 1)
    part_w = (W // area_ratio) * (part_level + 1)
    new_map = map[:, :, H // 2 - part_h // 2: H // 2 + part_h // 2, W // 2 - part_w // 2: W // 2 + part_w // 2]
    return new_map


def vis(img, name=''):
    import matplotlib.pyplot as plt
    img = img.cpu().numpy().astype(np.uint8)
    img[img == 0] = 255
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig('test_{}.png'.format(name))


def vis_map(rmap, name=''):
    import matplotlib.pyplot as plt
    rmap_idx = rmap.argmax(0).cpu().numpy()

    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
    img = colors[rmap_idx].astype(np.uint8)

    plt.imshow(img)
    plt.savefig('test_{}.png'.format(name))


def vis_map_contrastive(rmap1, rmap2, name=''):
    import matplotlib.pyplot as plt
    rmap1_idx = rmap1.argmax(0).cpu().numpy()
    rmap2_idx = rmap2.argmax(0).cpu().numpy()

    # rmap1_idx[rmap1_idx < 3] = 0
    # rmap2_idx[rmap2_idx < 3] = 2

    img_idx = np.full_like(rmap1_idx, 3)

    diff = (rmap2_idx < 3) ^ (rmap1_idx < 3)
    diff1 = (rmap1_idx < 3) & diff
    diff2 = (rmap2_idx < 3) & diff

    img_idx[diff1] = 2
    img_idx[diff2] = 0

    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
    img = colors[img_idx].astype(np.uint8)

    plt.imshow(img)
    plt.savefig('contrative.png'.format(name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate nuScenes local HD Map Construction Results.')
    # parser.add_argument('--result_path', type=str,
    #                     default='./work_dirs/results/results_nuscence.pkl')
    parser.add_argument('--result_path', type=str,
                        default='./work_dirs/gen_aug_larger/check_Eval.pkl')
    # parser.add_argument('--ann_file', type=str,
    #                     default='/mnt/datasets/nuScenes/nuScences_map_debug_infos_val.pkl')
    # parser.add_argument('--dataroot', type=str,
    #                     default='/mnt/datasets/nuScenes/')
    parser.add_argument('--ann_file', type=str,
                        default='/public/MARS/datasets/nuScenes/nuScences_map_trainval_infos_val.pkl')
    parser.add_argument('--dataroot', type=str,
                        default='/public/MARS/datasets/nuScenes')
    parser.add_argument('--bsz', type=int, default=128)
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--eval_set', type=str, default='mini_val',
                        choices=['train', 'val', 'test', 'mini_train', 'mini_val'])
    parser.add_argument('--thickness', type=int, default=5)
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('--CD_threshold', type=int, default=10)
    parser.add_argument("--xbound", nargs=3, type=float,
                        default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float,
                        default=[-15.0, 15.0, 0.15])

    args = parser.parse_args()

    args = vars(args)
    args['result_path'] = 5
    # args['result_path'] = '/home/dcz/liuyicheng_temp/marsmap/model_file/rpv_convex_mp_v2_dg/results_nuscence.pkl'
    # args['result_path'] = '/home/dcz/liuyicheng_temp/marsmap/model_file/hdmapnet_lidar/results_nuscence.pkl'
    # args['result_path'] = '/home/dcz/liuyicheng_temp/marsmap/work_dirs/hdmapnet_sh/withgt.pkl'
    # args['result_path'] = '/home/dcz/liuyicheng_temp/marsmap/work_dirs/hdmapnet_sh/hdmap_reproduce.pkl'
    args['result_path'] = '/home/dcz/liuyicheng_temp/marsmap/work_dirs/hdmapnet_sh/raster_reproduce.pkl'
    pprint(get_val_info(args))

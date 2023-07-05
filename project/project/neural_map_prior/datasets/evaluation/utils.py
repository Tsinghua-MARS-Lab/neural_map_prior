import mmcv
import numpy as np
import torch


def get_pos_idx(pos_msk):
    '''
        Arg:
            pos_msk: bxk
        Return:
            remain_idx: bxn
            remain_idx_mask: bxn
    '''

    valid_len = pos_msk.sum(-1)  # b
    max_len = valid_len.max()

    remain_idx = torch.argsort(pos_msk.type(
        torch.float32), -1, descending=True)  # bxnR

    remain_idx_mask = torch.arange(
        0, max_len, device=pos_msk.device)  # k
    remain_idx_mask = remain_idx_mask[None] < valid_len[:, None]  # bxk

    remain_idx = remain_idx[:, :max_len]  # bxk

    return remain_idx, remain_idx_mask


class CaseLogger:

    def __init__(self, model_name='hdmap_detr', predictions=None, infos=None, version=None):

        self.case_criterion = []
        edges = np.linspace(0, 1, 11)
        for i in range(1, len(edges)):
            cri = {}
            cri['range'] = (edges[i - 1], edges[i])
            cri['max_case'] = 3
            cri['case_num'] = 0
            cri['case'] = []
            self.case_criterion.append(cri)
        self.model_name = model_name

        # info
        self.version = version
        self.infos = infos
        self.predictions = predictions

    def save_packed_data(self, criterion, gt_maps, pred_map, idxes, save=True):
        '''
            find typical case and save
        '''
        for i, cri in enumerate(criterion):

            case_idx = self.check_typicality(cri)
            if not case_idx:
                continue

            idx = idxes[i]

            # load image
            filename = self.infos[idx]['img_filenames']
            imgs = [mmcv.imread(name, 'unchanged') for name in filename]

            data = {
                'img': imgs,
                'img_metas': self.infos[idx],
                'gts': gt_vector,
                'gt_map': gt_maps[i],
                'preds': self.prediction['results'][self.infos[idx]['token']]['vectors'],
            }

            if save:
                mmcv.dump(
                    data, f'./debug_img/{self.model_name}_cri_{i}_train_case{case_idx}_{cri}.pkl')
            else:
                self.case_criterion[i]['case'].append(data)

    def check_typicality(self, x):
        ''' Check the x is still neccessary for the logger'''
        flag = False
        for i, cri in enumerate(self.case_criterion):
            edges = cri['range']
            if edges[0] <= x and x < edges[1]:
                if cri['case_num'] >= cri['max_case']:
                    break

                self.case_criterion[i]['case_num'] += 1
                flag = self.case_criterion[i]['case_num']
                if cri['case_num'] == cri['max_case']:
                    print('Interval [{},{}) is full!'.format(edges[0], edges[1]))

        return flag

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # if (x1.shape[-2:] != x2.shape[-2:]):
        #    x1 = F.interpolate(x1, size=(x2.shape[-2:]))

        # x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


@HEADS.register_module()
class BevEncode(nn.Module):
    def __init__(self,
                 inC,
                 outC,
                 instance_seg=True,
                 embedded_dim=16,
                 direction_pred=True,
                 direction_dim=37,
                 return_feature=False):

        super(BevEncode, self).__init__()
        # trunk = resnet18(pretrained=False, zero_init_residual=True)
        # self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        # self.bn1 = trunk.bn1
        # self.relu = trunk.relu

        # self.layer1 = trunk.layer1
        # self.layer2 = trunk.layer2
        # self.layer3 = trunk.layer3

        # self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        self.return_feature = return_feature

        self.instance_seg = instance_seg
        if instance_seg:
            # self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
            # self.up1_embedded = Up(256, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            # self.up1_direction = Up(64 + 256, 256, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )

    def forward(self, bev_feature: list):
        x = bev_feature[0]

        # if (x.shape[-2:] != (200,400)):
        #    x = F.interpolate(x, size=(200,400))

        # x2, x1 = self.encoder(x)

        # x = self.up1(x2, x1)
        x_seg = self.up2(x)

        if self.instance_seg:
            # x_embedded = self.up1_embedded(x2, x1)
            x_embedded = x
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None

        if self.direction_pred:
            # x_direction = self.up1_embedded(x2, x1)
            x_direction = x
            x_direction = self.up2_direction(x_direction)
        else:
            x_direction = None

        out = dict(
            preds_map=x_seg,
            embedded=x_embedded,
            direction=x_direction,
        )

        if self.return_feature:
            return out, x2

        return out

    def encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)

        return x2, x1

    # def post_process(self, preds_dict, vectors, tokens):

    #     pred_maps = preds_dict['preds_map']
    #     embedded = preds_dict['embedded']
    #     direction = preds_dict['direction']
    #     batch_size = pred_maps.size(0)

    #     # since in liqi‘s code 2 is ped and 1 is divider
    #     real_label = [1,0,2]

    #     ret_list = []
    #     for batch_idx in range(batch_size):

    #         coords, confidences, line_types = \
    #             vectorize(pred_maps[batch_idx], embedded[batch_idx], direction[batch_idx], angle_class=36)

    #         # since in liqi‘s code 2 is ped and 1 is divider which is
    #         # revserse with the definition in evaluation
    #         pred_map = F.softmax(pred_maps[batch_idx][(0,2,1,3),], dim=0)
    #         # if True:
    #         #     vis_map(pred_map,True,'pred_map')
    #         #     import matplotlib.pyplot as plt
    #         #     for i in coords:
    #         #         i = i.astype(np.float)
    #         #         plt.plot(i[:,0],i[:,1],'-',lw=3)
    #         #     plt.savefig('./hdmapnet/{}.png'.format(tokens[batch_idx]))
    #         #     plt.close()

    #         max_num = pred_map.max(0)[0]
    #         onehot_preds = pred_map == max_num

    #         # filter the background
    #         onehot_preds = onehot_preds[1:].detach().cpu().numpy()
    #         pred_map = pred_map[1:].detach().cpu().numpy()

    #         lines, scores, labels = [], [], []
    #         for i in range(len(coords)):
    #             l = coords[i].astype(np.float)/np.array((400,200))
    #             l = l*2 - 1
    #             lines.append(l)
    #             scores.append(confidences[i])
    #             label = real_label[line_types[i]]
    #             labels.append(label)

    #         # ret_dict_single = {
    #         #     'token': tokens[i],
    #         #     'pred_map': pred_map,
    #         # }

    #         ret_dict_single = {
    #             'token': tokens[batch_idx],
    #             'scores': np.array(scores),
    #             'labels':np.array(labels),
    #             'lines': lines,
    #             'nline': len(coords),
    #         }

    #         vec_lines_sample = vectors[batch_idx]
    #         num_gt = len(vec_lines_sample)
    #         lines_list = []
    #         labels_list = []
    #         for _l, (vec_lines) in enumerate(vec_lines_sample):
    #             lines_list.append(vec_lines[0]*2-1)
    #             labels_list.append(vec_lines[2])

    #         ret_dict_single['groundTruth'] = {
    #             'token': tokens[batch_idx],
    #             'nline': num_gt,
    #             'labels': labels_list,
    #             'lines': lines_list,
    #         }

    #         ret_list.append(ret_dict_single)

    #     return ret_list

    def post_process(self, preds_dict, vectors, tokens):

        pred_maps = preds_dict['preds_map']
        batch_size = pred_maps.size(0)

        ret_list = []
        for i in range(batch_size):
            pred_map = pred_maps[i]
            # since in liqi‘s code 2 is ped and 1 is divider which is
            # revserse with the definition in evaluation
            pred_map = F.softmax(pred_map[(0, 2, 1, 3),], dim=0)
            max_num = pred_map.max(0)[0]

            pred_map = pred_map == max_num
            # filter the background
            pred_map = pred_map[1:].detach().cpu().numpy()

            # save_dir = '/home/xiongx/repository/marsmap/results_cvpr/lss_final'
            save_dir = '/home/xiongx/repository/marsmap/results_cvpr/bevformer_fianl_night'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(pred_map.astype(np.uint8), os.path.join(save_dir, tokens[i]))

            ret_dict_single = {
                'token': tokens[i],
                'pred_map': pred_map.astype(np.uint8),
            }

            ret_list.append(ret_dict_single)

        return ret_list


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def vis_map(rmap, flip=False, name=''):
    import matplotlib.pyplot as plt
    rmap_idx = rmap.argmax(0).cpu().numpy()

    # colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])
    colors = np.array([[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
    img = colors[rmap_idx].astype(np.uint8)

    # if flip:
    #     img = rotate(img,180)
    #     img = img[::-1,:,:]

    plt.imshow(img)
    plt.savefig('test_{}.png'.format(name))

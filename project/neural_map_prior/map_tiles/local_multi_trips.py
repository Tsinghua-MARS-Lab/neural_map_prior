import os
import shutil
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from project.neural_map_prior.map_tiles.lane_render import load_nusc_data_infos, load_nusc_data_cities

# vis_token_list = [
#     'bbe760f774824fb5a34b7999dcd999b3',
#     '9bb6116e8f194294bf5c2107c42acc1a',
#     'bfbec01aed26439ab0b1442b10a75a16',
#     # 'a027c6b262db48f79a281d674775a778',
#     # '95682f8329cf487aa061d968b08f80ea',
# ]

vis_token_list = [
    # 'bbe760f774824fb5a34b7999dcd999b3'
    # '9bb6116e8f194294bf5c2107c42acc1a'
    'bfbec01aed26439ab0b1442b10a75a16'
]


def get_relative_angle(trans, ref_trans):
    rotation_matrix = np.linalg.inv(ref_trans[:3, :3]) @ trans[:3, :3]
    v = np.dot(rotation_matrix, np.array([1, 0, 0]))
    yaw = np.arctan2(v[1], v[0])
    rotation_in_degrees = -yaw / np.pi * 180
    return rotation_in_degrees


# def find_max_radius(cur_seq, cur_frame, his_trips_seq, sample_tokens):
#     max_radius = 0
#     cur_token = sample_tokens[cur_seq][cur_frame]
#     cur_sample_id = token2sample_id(cur_token, sample_tokens)
#     cur_info = nusc_data_infos_val[cur_sample_id]
#     cur_trans = gen_matrix(
#         cur_info['ego2global_rotation'],
#         cur_info['ego2global_translation']
#     )
#     cur_geo = cur_trans[:2, 3]
#     for his_seq in his_trips_seq:
#         his_frame_list = sample_tokens[his_seq]
#
#         for his_token in his_frame_list:
#             his_sample_id = token2sample_id(his_token, sample_tokens)
#             his_info = nusc_data_infos_val[his_sample_id]
#             his_trans = gen_matrix(
#                 his_info['ego2global_rotation'],
#                 his_info['ego2global_translation']
#             )
#             his_geo = his_trans[:2, 3]
#             raius = np.max(np.abs(cur_geo - his_geo))
#             if raius > max_radius:
#                 max_radius = raius
#     return max_radius


def find_max_radius_hw(cur_seq, cur_frame, his_trips_seq, sample_tokens):
    max_radius_h = 0
    max_radius_w = 0
    cur_token = sample_tokens[cur_seq][cur_frame]
    cur_sample_id = token2sample_id(cur_token, sample_tokens)
    cur_info = nusc_data_infos_val[cur_sample_id]
    cur_trans = gen_matrix(
        cur_info['ego2global_rotation'],
        cur_info['ego2global_translation']
    )
    cur_geo = cur_trans[:2, 3]
    for his_seq in his_trips_seq:
        his_frame_list = sample_tokens[his_seq]
        for his_token in his_frame_list:
            his_sample_id = token2sample_id(his_token, sample_tokens)
            his_info = nusc_data_infos_val[his_sample_id]
            his_trans = gen_matrix(
                his_info['ego2global_rotation'],
                his_info['ego2global_translation']
            )
            his_geo = his_trans[:2, 3]
            radius_h = np.abs(cur_geo[0] - his_geo[0])
            radius_w = np.abs(cur_geo[1] - his_geo[1])
            if radius_h > max_radius_h:
                max_radius_h = radius_h
            if radius_w > max_radius_w:
                max_radius_w = radius_w
    return max_radius_h, max_radius_w


def token2seq_frame(token, sample_tokens):
    for i, tokens in enumerate(sample_tokens):
        for j, t in enumerate(tokens):
            if t == token:
                return i, j


def seq_frame2token(seq, frame, sample_tokens):
    return sample_tokens[seq][frame]


def token2sample_id(token, sample_tokens):
    count = 0
    for i, tokens in enumerate(sample_tokens):
        for j, t in enumerate(tokens):
            if t == token:
                return count
            count += 1


def get_bev_coords(bev_bound_w, bev_bound_h, bev_w, bev_h, type_flag='numpy'):
    '''
    Args:
        bev_bound_w (tuple:2):
        bev_bound_h (tuple:2):
        bev_w (int):
        bev_h (int):

    Returns: (bev_h, bev_w, 4)

    '''

    if type_flag == 'torch':
        sample_coords = torch.stack(torch.meshgrid(
            torch.linspace(bev_bound_w[0], bev_bound_w[1], int(bev_w), dtype=torch.float32),
            torch.linspace(bev_bound_h[0], bev_bound_h[1], int(bev_h), dtype=torch.float32)
        ), axis=2).transpose(1, 0)
        assert sample_coords.shape[0] == bev_h, sample_coords.shape[1] == bev_w
        zeros = torch.zeros((bev_h, bev_w, 1), dtype=sample_coords.dtype)
        ones = torch.ones((bev_h, bev_w, 1), dtype=sample_coords.dtype)
        sample_coords = torch.cat([sample_coords, zeros, ones], dim=-1)
    elif type_flag == 'numpy':
        sample_coords = np.meshgrid(np.linspace(bev_bound_w[0], bev_bound_w[1], int(bev_w), dtype=np.float32),
                                    np.linspace(bev_bound_h[0], bev_bound_h[1], int(bev_h), dtype=np.float32))
        sample_coords = np.stack(sample_coords, axis=2)
        assert sample_coords.shape[0] == bev_h, sample_coords.shape[1] == bev_w
        zeros = np.zeros((bev_h, bev_w, 1), dtype=sample_coords.dtype)
        ones = np.ones((bev_h, bev_w, 1), dtype=sample_coords.dtype)
        sample_coords = np.concatenate([sample_coords, zeros, ones], axis=-1)
    return sample_coords


def get_pic_loc(bev_feature,
                pad_bev_feature_int,
                pad_bev_feature,
                ego2ego,
                global_bound_h,
                global_bound_w,
                real_h=30,
                real_w=60,
                bev_h=200,
                bev_w=400,
                global_factor=1,
                type_flag='numpy'):
    '''
    Args:
        bev_feature (B, bev_h, bev_w, C):
        pad_bev_feature (B, bev_h, bev_w, C):
        ego2ego (4, 4):
        real_h (int):
        real_w (int):
    Returns: (B, bev_h, bev_w, C)

    '''
    # device = bev_feature.device
    # B, bev_h, bev_w, C = bev_feature.size()

    bev_bound_h, bev_bound_w = \
        [(-row[0] / 2 + row[0] / row[1] / 2, row[0] / 2 - row[0] / row[1] / 2)
         for row in ((real_h, bev_h), (real_w, bev_w))]
    # bev_bound_h, bev_bound_w = \
    #     [(-row[0] / 2, row[0] / 2)
    #      for row in ((real_h, bev_h), (real_w, bev_w))]
    grid_len_h = real_h / (bev_h / global_factor)
    grid_len_w = real_w / (bev_w / global_factor)

    bev_coords = get_bev_coords(bev_bound_w, bev_bound_h, bev_w, bev_h, type_flag)

    if type_flag == 'torch':
        ego2ego = bev_coords.new_tensor(ego2ego)
        bev_coords = bev_coords.reshape(-1, 4).permute(1, 0)
    elif type_flag == 'numpy':
        bev_coords = bev_coords.reshape(-1, 4).transpose(1, 0)

    trans_bev_coords = ego2ego @ bev_coords
    ##########################
    trans_bev_coords[0] += pad_bev_feature.shape[1] / 2 * (real_h / bev_h)
    trans_bev_coords[1] += pad_bev_feature.shape[2] / 2 * (real_w / bev_w)

    bev_coord_w = trans_bev_coords[0, :]
    bev_coord_h = trans_bev_coords[1, :]

    if type_flag == 'torch':
        bev_index_w = torch.floor((bev_coord_w - global_bound_w[0]) / grid_len_w).to(torch.int64)
        bev_index_h = torch.floor((bev_coord_h - global_bound_h[0]) / grid_len_h).to(torch.int64)
    elif type_flag == 'numpy':
        bev_index_w = np.floor((bev_coord_w - global_bound_w[0]) / grid_len_w).astype(np.int64)
        bev_index_h = np.floor((bev_coord_h - global_bound_h[0]) / grid_len_h).astype(np.int64)

    bev_index_w = bev_index_w.reshape(bev_h, bev_w)
    bev_index_h = bev_index_h.reshape(bev_h, bev_w)

    return bev_index_w, bev_index_h


def get_coords_resample(bev_feature,
                        pad_bev_feature_int,
                        pad_bev_feature,
                        ego2ego,
                        global_bound_h,
                        global_bound_w,
                        real_h=30,
                        real_w=60,
                        bev_h=200,
                        bev_w=400,
                        reverse=False,
                        flag='replace',
                        traversal_id=None,
                        global_factor=1,
                        bev_feature_int=None,
                        view_info=False,
                        type_flag='numpy'):
    '''
    Args:
        bev_feature (B, bev_h, bev_w, C):
        pad_bev_feature (B, bev_h, bev_w, C):
        ego2ego (4, 4):
        real_h (int):
        real_w (int):
    Returns: (B, bev_h, bev_w, C)

    '''
    # device = bev_feature.device
    # B, bev_h, bev_w, C = bev_feature.size()

    bev_bound_h, bev_bound_w = \
        [(-row[0] / 2 + row[0] / row[1] / 2, row[0] / 2 - row[0] / row[1] / 2)
         for row in ((real_h, bev_h), (real_w, bev_w))]
    # bev_bound_h, bev_bound_w = \
    #     [(-row[0] / 2, row[0] / 2)
    #      for row in ((real_h, bev_h), (real_w, bev_w))]
    grid_len_h = real_h / (bev_h / global_factor)
    grid_len_w = real_w / (bev_w / global_factor)

    bev_coords = get_bev_coords(bev_bound_w, bev_bound_h, bev_w, bev_h, type_flag)

    if type_flag == 'torch':
        ego2ego = bev_coords.new_tensor(ego2ego)
        bev_coords = bev_coords.reshape(-1, 4).permute(1, 0)
    elif type_flag == 'numpy':
        bev_coords = bev_coords.reshape(-1, 4).transpose(1, 0)

    trans_bev_coords = ego2ego @ bev_coords
    ##########################
    trans_bev_coords[0] += pad_bev_feature.shape[1] / 2 * (real_h / bev_h)
    trans_bev_coords[1] += pad_bev_feature.shape[2] / 2 * (real_w / bev_w)

    bev_coord_w = trans_bev_coords[0, :]
    bev_coord_h = trans_bev_coords[1, :]

    if type_flag == 'torch':
        bev_index_w = torch.floor((bev_coord_w - global_bound_w[0]) / grid_len_w).to(torch.int64)
        bev_index_h = torch.floor((bev_coord_h - global_bound_h[0]) / grid_len_h).to(torch.int64)
    elif type_flag == 'numpy':
        bev_index_w = np.floor((bev_coord_w - global_bound_w[0]) / grid_len_w).astype(np.int64)
        bev_index_h = np.floor((bev_coord_h - global_bound_h[0]) / grid_len_h).astype(np.int64)

    bev_index_w = bev_index_w.reshape(bev_h, bev_w)
    bev_index_h = bev_index_h.reshape(bev_h, bev_w)

    bev_coord_mask = \
        (global_bound_w[0] <= bev_coord_w) & (bev_coord_w < global_bound_w[1]) & \
        (global_bound_h[0] <= bev_coord_h) & (bev_coord_h < global_bound_h[1])
    bev_coord_mask = bev_coord_mask.reshape(bev_h, bev_w)

    if type_flag == 'torch':
        index_h, index_w = torch.where(bev_coord_mask.reshape(bev_h, bev_w))
    elif type_flag == 'numpy':
        index_h, index_w = np.where(bev_coord_mask.reshape(bev_h, bev_w))

    # overlap_feats = bev_feature[:, index_h, index_w, :]
    # if type_flag == 'torch':
    #     pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] = overlap_feats.to('cpu')
    # elif type_flag == 'numpy':
    #     pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] = overlap_feats

    if pad_bev_feature_int is not None:
        pad_bev_feature_int[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] += 1
    if reverse:
        overlap_feats = pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :]
        if type_flag == 'torch':
            overlap_feats = overlap_feats.to(bev_feature.device).clone()
        bev_feature[:, index_h, index_w, :] = overlap_feats

        if bev_feature_int is not None:
            bev_feature_int[:, index_h, index_w, :] = \
                pad_bev_feature_int[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :].to(
                    bev_feature_int.device).clone()
    else:
        if pad_bev_feature is not None:
            overlap_feats = bev_feature[:, index_h, index_w, :]
            if type_flag == 'torch':
                overlap_feats = overlap_feats.to('cpu')
            if flag == 'add':
                pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] += overlap_feats
            elif flag == 'replace':
                pad_bev_feature[:, bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] = overlap_feats


def gen_matrix(ego2global_rotation, ego2global_translation):
    rotation_xyz = np.roll(ego2global_rotation, shift=-1)
    trans = np.eye(4)
    trans[:3, 3] = ego2global_translation
    trans[:3, :3] = R.from_quat(rotation_xyz).as_matrix()
    return trans


def load_history_trip_info(root=None, file_name=None):
    with open(os.path.join(root, 'sample_tokens.pkl'), 'rb') as f:
        sample_tokens = pkl.load(f)
    with open(os.path.join(root, 'sample_poses_array.pkl'), 'rb') as f:
        sample_poses_array = pkl.load(f)
    with open(os.path.join(root, 'sample_trans_array.pkl'), 'rb') as f:
        sample_trans_array = pkl.load(f)
    with open(os.path.join(root, f'{file_name}.pkl'), 'rb') as f:
        hdmap_history_dict = pkl.load(f)

    return sample_tokens, sample_poses_array, sample_trans_array, hdmap_history_dict


def seq_id2timestamp(seq_id, nusc, sample_tokens):
    return nusc.get('sample', sample_tokens[seq_id][0])['timestamp']


if __name__ == '__main__':
    result_root = '/home/xiongx/repository/marsmap/vis_results/bevformer_highest_gm'
    save_root = '/home/xiongx/repository/marsmap/vis_results/multi_trips_map_eighth'

    bev_attribute = {
        'real_h': 30,
        'real_w': 60,
        'bev_h': 200,
        'bev_w': 400
    }
    canvas_dim = 3
    dpi = 3500

    dataset = 'val'
    sample_tokens, \
        sample_poses_array, \
        sample_trans_array, \
        hdmap_history_dict = \
        load_history_trip_info(
            root='/home/xiongx/repository/marsmap/xiongxuan/traversals_processing/hdmap_history_loc_val_h60_w30_thr0',
            file_name=f'hdmap_history_{dataset}_30_60_1'
        )

    # nusc = NuScenes(version='v1.0-trainval', dataroot='/public/MARS/datasets/nuScenes/', verbose=True)
    # seq2timestamp = dict()
    # for seq_id in range(len(sample_tokens)):
    #     seq2timestamp[seq_id] = seq_id2timestamp(seq_id, nusc, sample_tokens)
    # with open(os.path.join('.', f'{dataset}_seq2timestamp.pkl'), 'wb') as f:
    #     pkl.dump(seq2timestamp, f)

    with open(os.path.join('/oldhome/xiongx/repository/marsmap/xiongxuan', f'{dataset}_seq2timestamp.pkl'), 'rb') as f:
        seq2timestamp = pkl.load(f)

    nusc_data_infos_val = load_nusc_data_infos(dataset)
    val_data_infos = load_nusc_data_infos(dataset)
    val_data_cities = load_nusc_data_cities(dataset, root=None)

    num_overlap_trips = list()
    for info in nusc_data_infos_val:
        cur_seq, cur_frame = token2seq_frame(info['token'], sample_tokens)
        try:
            trips_list = hdmap_history_dict[f'{cur_seq}_{cur_frame}']
        except:
            num_overlap_trips.append(0)
            continue
        num_overlap_trips.append(len(trips_list))

    # plt.hist(num_overlap_trips, bins=100)
    # plt.savefig(os.path.join(save_root, f'{dataset}_trip_hist.png'))

    sort_infos = np.argsort(num_overlap_trips)[::-1]
    for i, num_info in tqdm(enumerate(sort_infos)):
        # if i < 121:
        #     continue
        # 500
        info = nusc_data_infos_val[num_info]
        if info['token'] not in vis_token_list:
            continue

        if num_info % 5:
            continue

        cur_seq, cur_frame = token2seq_frame(info['token'], sample_tokens)
        cur_trans = gen_matrix(
            info['ego2global_rotation'],
            info['ego2global_translation']
        )
        try:
            trips_list = hdmap_history_dict[f'{cur_seq}_{cur_frame}']
        except:
            continue

        his_trips_seq = [trip[0] for trip in trips_list]
        his_trips_seq.append(cur_seq)
        his_trips_seq_timestamp = [seq2timestamp[seq] for seq in his_trips_seq]
        his_trips_seq = [his_trips_seq[ind] for ind in np.argsort(his_trips_seq_timestamp)]
        print(np.argsort([seq2timestamp[seq] for seq in his_trips_seq]))
        print(f'this is {num_info}th sample, {len(his_trips_seq)}th trips')

        # canvas_size = find_max_radius_hw(cur_seq, cur_frame, his_trips_seq, sample_tokens)
        max_radius_h, max_radius_w = find_max_radius_hw(cur_seq, cur_frame, his_trips_seq, sample_tokens)
        canvas_size = [np.int64(max_radius_w / 0.15 + 600 + 1 + 270 + 20) * 2,
                       np.int64(max_radius_h / 0.15 + 600 + 1 + 270 + 20) * 2]
        # canvas_size = [np.int64(canvas_size / 0.15) * 2, np.int64(canvas_size / 0.15 + 600) * 2]

        count = 0
        canvas = np.zeros((1, *canvas_size, canvas_dim))

        for his_seq in his_trips_seq:
            for his_frame, his_token in enumerate(sample_tokens[his_seq]):
                his_sample_id = token2sample_id(his_token, sample_tokens)
                his_info = nusc_data_infos_val[his_sample_id]
                his_trans = gen_matrix(
                    his_info['ego2global_rotation'],
                    his_info['ego2global_translation']
                )
                his_bev_feat = torch.load(os.path.join(result_root, his_token)).astype(np.int32)
                his_bev_feat = his_bev_feat.transpose(1, 2, 0)[np.newaxis]

                ego2ego_trans = np.linalg.inv(cur_trans) @ his_trans

                get_coords_resample(
                    his_bev_feat,
                    None,
                    canvas,
                    ego2ego_trans,
                    global_bound_h=(0, canvas.shape[2]),
                    global_bound_w=(0, canvas.shape[1]),
                    real_h=bev_attribute['real_h'],
                    real_w=bev_attribute['real_w'],
                    bev_h=bev_attribute['bev_h'],
                    bev_w=bev_attribute['bev_w'],
                    flag='replace'
                )

                # plt.imshow(
                #     arrow_img_rotate,
                #     extent=[
                #         loc_center_w[4999] + -35,
                #         loc_center_w[4999] + 35,
                #         loc_center_h[40000] + -35,
                #         loc_center_h[40000] + 35
                #     ],
                #     alpha=0.5
                # )
                # plt.show()

                semantic_show = np.full(
                    shape=(3, canvas.shape[1],
                           canvas.shape[2]),
                    fill_value=255, dtype=np.uint8)
                semantic_show[:, canvas[0, ..., 0].astype(np.int32) == 1] = \
                    np.array([97, 135, 178], dtype=np.uint8).reshape(3, 1)
                semantic_show[:, canvas[0, ..., 1].astype(np.int32) == 1] = \
                    np.array([178, 97, 103], dtype=np.uint8).reshape(3, 1)
                semantic_show[:, canvas[0, ..., 2].astype(np.int32) == 1] = \
                    np.array([107, 156, 123], dtype=np.uint8).reshape(3, 1)

                # plt.figure(figsize=(4, 2), dpi=1500)
                plt.figure(figsize=(2, 2), dpi=dpi)
                plt.imshow(semantic_show.transpose((1, 2, 0)))
                plt.axis('off')
                plt.show()

                plt.xlim(0, canvas.shape[1])
                plt.ylim(0, canvas.shape[2])
                arrow = Image.open('icon/arrow.png')
                rotation_in_degrees = get_relative_angle(his_trans, cur_trans)
                arrow_img_rotate = arrow.rotate(rotation_in_degrees + 90, expand=True)
                loc_center_w, loc_center_h = get_pic_loc(
                    his_bev_feat,
                    None,
                    canvas,
                    ego2ego_trans,
                    global_bound_h=(0, canvas.shape[2]),
                    global_bound_w=(0, canvas.shape[1]),
                    real_h=bev_attribute['real_h'],
                    real_w=bev_attribute['real_w'],
                    bev_h=bev_attribute['bev_h'],
                    bev_w=bev_attribute['bev_w'])

                plt.imshow(
                    arrow_img_rotate,
                    extent=[
                        loc_center_h[100, 200] + -35,
                        loc_center_h[100, 200] + 35,
                        loc_center_w[100, 200] + -35,
                        loc_center_w[100, 200] + 35
                    ],
                    alpha=1
                )
                plt.show()
                loc_center_h_x = np.stack([
                    loc_center_h[0, 0], loc_center_h[-1, 0],
                    loc_center_h[-1, -1], loc_center_h[0, -1],
                    loc_center_h[0, 0]])
                loc_center_w_y = np.stack([
                    loc_center_w[0, 0], loc_center_w[-1, 0],
                    loc_center_w[-1, -1], loc_center_w[0, -1],
                    loc_center_w[0, 0]])
                plt.plot(loc_center_h_x, loc_center_w_y,
                         'b' + '--', linewidth=0.6, alpha=0.8)
                plt.show()

                save_path = os.path.join(save_root, info['token'] + '-' + f'{cur_seq}' + '-' + f'{len(his_trips_seq)}')
                os.makedirs(save_path, exist_ok=True)

                imname = f'{save_path}/{count}_{his_seq}_{len(trips_list)}.jpg'
                print('saving', imname)
                plt.savefig(imname)
                plt.close()
                count += 1

                # post_fix = '_surr'
                # os.makedirs(save_path + post_fix, exist_ok=True)
                # shutil.copyfile(f'normal_with_pred/{his_seq}_{his_frame}_{his_token}.png',
                #                 f'{save_path}{post_fix}/{count}_{his_seq}_{len(trips_list)}.jpg')

                # plt.figure(figsize=(3, 1), dpi=dpi)
                # plt.subplot(1, 2, 1)
                # plt.axis('off')
                # surr_images = Image.open(f'all_with_pred/{his_seq}_{his_frame}_{his_token}.png')
                # plt.imshow(surr_images)
                # plt.show()
                #
                # plt.subplot(1, 2, 2)
                # plt.axis('off')
                # global_map = Image.open(imname)
                # plt.imshow(global_map)
                # plt.show()
                #
                # os.remove(imname)
                # print('saving', imname.replace('.jpg', '_surr.jpg'))
                # plt.savefig(imname.replace('.jpg', '_surr.jpg'))

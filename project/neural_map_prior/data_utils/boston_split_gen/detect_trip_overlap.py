import argparse
import os
import pickle as pkl

import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from rotate_iou import rotate_iou_gpu_eval


def get_sample_token(nusc, scene):
    sample_token_list = [scene['first_sample_token']]
    while sample_token_list[-1] != scene['last_sample_token']:
        sample_token_list.append(nusc.get('sample', sample_token_list[-1])['next'])
    return sample_token_list


def get_trans_from_record(ego_pose_record):
    rotation = ego_pose_record['rotation'][1:] + \
               [ego_pose_record['rotation'][0]]
    oxts_info = [x for x in ego_pose_record['translation']] + \
                [x for x in R.from_quat(rotation).as_euler('xyz')]
    trans = np.eye(4)
    trans[:3, 3] = oxts_info[:3]
    trans[:3, :3] = R.from_euler('xyz', oxts_info[3:]).as_matrix()
    rotation_matrix2 = Quaternion(ego_pose_record['rotation']).rotation_matrix
    assert ((trans[:3, :3] - rotation_matrix2) > 1e-4).sum() == 0, \
        (trans, rotation_matrix2)
    return trans.astype(np.float32), oxts_info


def get_map_angle(ego2global_rotation):
    rotation = Quaternion(ego2global_rotation)
    patch_angle = quaternion_yaw(rotation) / np.pi * 180
    return patch_angle


def get_sample_pose(nusc, token_list):
    pose_list = []
    trans_list = []
    for sample_token in token_list:
        lidar_token = nusc.get('sample', sample_token)['data']['LIDAR_TOP']
        lidar_record = nusc.get('sample_data', lidar_token)
        ego_pose_record = nusc.get("ego_pose", lidar_record["ego_pose_token"])
        trans_matrix, oxts_info = get_trans_from_record(ego_pose_record)
        trans_list.append(trans_matrix)
        oxts_info.append(get_map_angle(ego_pose_record['rotation']))
        pose_list.append(oxts_info)
    return pose_list, trans_list


def filter_scenes(nusc, split_scenes, in_boston=False):
    new_split_scenes = []
    for scene in nusc.scene:
        if in_boston:
            if scene['name'] in split_scenes and \
                    'boston' in nusc.get('log',
                                         scene['log_token'])['location']:
                new_split_scenes.append(scene)
        else:
            if scene['name'] in split_scenes:
                new_split_scenes.append(scene)
    return new_split_scenes


def find_hdmap_history(args):
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    split_scenes_name = create_splits_scenes()[args.hist_set]
    split_scenes = filter_scenes(nusc, split_scenes_name, in_boston=False)
    split_scenes_loc = [nusc.get('log', scene['log_token'])['location'] for scene in split_scenes]
    
    sample_tokens = [get_sample_token(nusc, scene) for scene in split_scenes]
    sample_tokens = [
        [
            token for i, token in enumerate(tokens) if i % args.sample_interval == 0
        ]
        for tokens in sample_tokens
    ]
    
    sample_poses_trans = [get_sample_pose(nusc, token_list) for token_list in sample_tokens]
    sample_poses_array = [np.array(poses) for (poses, trans) in sample_poses_trans]
    sample_trans_array = [np.array(trans) for (poses, trans) in sample_poses_trans]
    
    os.makedirs(args.save_dir_name, exist_ok=True)
    with open(os.path.join(args.save_dir_name, 'split_scenes.pkl'), 'wb') as f:
        pkl.dump(split_scenes, f)
    with open(os.path.join(args.save_dir_name, 'sample_tokens.pkl'), 'wb') as f:
        pkl.dump(sample_tokens, f)
    with open(os.path.join(args.save_dir_name, 'sample_poses_array.pkl'), 'wb') as f:
        pkl.dump(sample_poses_array, f)
    with open(os.path.join(args.save_dir_name, 'sample_trans_array.pkl'), 'wb') as f:
        pkl.dump(sample_trans_array, f)
    
    hdmap_height = args.box_height
    hdmap_width = args.box_width
    
    hdmap_history_dict = {}
    scene_length = len(sample_tokens)
    for i in tqdm(range(scene_length)):
        for j in range(len(sample_tokens[i])):
            for n in range(scene_length):
                if n == i:
                    continue
                if split_scenes_loc[i] != split_scenes_loc[n]:
                    continue
                
                if nusc.get('sample', sample_tokens[i][j])['timestamp'] - \
                        nusc.get('sample', sample_tokens[n][0])['timestamp'] < 0:
                    continue
                
                ori_rbbox = np.array(
                    [
                        sample_poses_array[i][j][0],
                        sample_poses_array[i][j][1],
                        hdmap_width,
                        hdmap_height,
                        sample_poses_array[i][j][-1],
                    ], dtype=np.float32
                )
                ori_rbbox = ori_rbbox[np.newaxis, :]
                ref_rbbox = np.vstack(
                    [
                        sample_poses_array[n][:, 0],
                        sample_poses_array[n][:, 1],
                        np.repeat(hdmap_width, sample_poses_array[n][:, 0].shape[0]),
                        np.repeat(hdmap_height, sample_poses_array[n][:, 0].shape[0]),
                        sample_poses_array[n][:, -1],
                    ]
                ).transpose()
                
                sequence_distance = \
                    np.linalg.norm(
                        ori_rbbox[:, :2] - ref_rbbox[:, :2], ord=2, axis=1)
                
                diagonal_len = np.sqrt((hdmap_width / 2) ** 2 + (hdmap_height / 2) ** 2) * 2
                if (sequence_distance < diagonal_len).sum() == 0:
                    continue
                
                iou = rotate_iou_gpu_eval(ori_rbbox, ref_rbbox)
                frame_id_lvl1 = np.where(iou >= args.iou_thr)[1]
                frame_id_lvl2 = np.where((iou > 1e-3) & (iou < args.iou_thr))[1]
                
                if len(frame_id_lvl1) > 0:
                    content_tuple = hdmap_history_dict.get(f'{i}_{j}', [])
                    content_tuple.append(
                        (
                            n,
                            frame_id_lvl1.tolist(),
                            iou[:, frame_id_lvl1],
                            frame_id_lvl2.tolist(),
                            iou[:, frame_id_lvl2]
                        )
                    )
                    hdmap_history_dict[f'{i}_{j}'] = content_tuple
    
    with open(
            os.path.join(
                args.save_dir_name,
                f'{args.save_file_name}_{args.hist_set}_' +
                f'{int(hdmap_width)}_{int(hdmap_height)}_' +
                f'{args.sample_interval}.pkl'),
            'wb') as f:
        pkl.dump(hdmap_history_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet history.')
    
    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='/public/MARS/datasets/nuScenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'])
    
    # set config
    # parser.add_argument('--hist_set', type=str, default='train')
    parser.add_argument('--hist_set', type=str, default='val')
    parser.add_argument("--sample_interval", type=int, default=1)
    
    # data config
    parser.add_argument('--box_height', type=float, default=60.0)
    parser.add_argument('--box_width', type=float, default=30.0)
    parser.add_argument("--iou_thr", type=float, default=0.00)
    
    # file config
    # parser.add_argument('--save_dir_name', type=str, default='hdmap_history_loc_train_h60_w30_thr0')
    parser.add_argument('--save_dir_name', type=str, default='trip_overlap_val_h60_w30_thr0')
    parser.add_argument('--save_file_name', type=str, default='trip_overlap_val_60_30_1')
    
    parse_args = parser.parse_args()
    find_hdmap_history(parse_args)

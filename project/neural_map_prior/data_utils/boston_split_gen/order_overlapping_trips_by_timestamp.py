import pickle as pkl

import numpy as np
from nuscenes.nuscenes import NuScenes

with open('./trip_overlap_val_h60_w30_thr0/sample_tokens.pkl', 'rb') as f:
    sample_tokens = pkl.load(f)
with open('./trip_overlap_val_h60_w30_thr0/trip_overlap_val_60_30_1.pkl', 'rb') as f:
    trip_overlap_val = pkl.load(f)


def seq_id2timestamp(seq_id, nusc, sample_tokens):
    return nusc.get('sample', sample_tokens[seq_id][0])['timestamp']


token2repeat_seq = {}
token2traversal_id = {}
for cur_seq_frame_id, overlap_trips in trip_overlap_val.items():
    cur_seq_id, cur_frame_id = cur_seq_frame_id.split('_')
    cur_seq_id = int(cur_seq_id)
    cur_frame_id = int(cur_frame_id)
    
    token = sample_tokens[cur_seq_id][cur_frame_id]
    ref_seq_list = [cur_seq_id]
    for i in range(len(overlap_trips)):
        ref_seq_list.append(overlap_trips[i][0])
    token2repeat_seq[token] = ref_seq_list
    token2traversal_id[token] = cur_seq_id

nusc = NuScenes(version='v1.0-trainval', dataroot='/dataroot/nuScenes/', verbose=True)
token2repeat_seq_sort_by_time = {}
for token, ref_seq_list in token2repeat_seq.items():
    ref_seq_list = np.array(ref_seq_list)
    ref_seq_timestamp = [seq_id2timestamp(seq_id, nusc, sample_tokens) for seq_id in ref_seq_list]
    ref_seq_list_sort = ref_seq_list[np.argsort(ref_seq_timestamp)].tolist()
    
    token2repeat_seq_sort_by_time[token] = {seq_id: ind for ind, seq_id in enumerate(ref_seq_list_sort)}
    oseq_id = token2traversal_id[token]
    token2traversal_id[token] = token2repeat_seq_sort_by_time[token][oseq_id]

print(len(token2traversal_id))
with open('val_token2traversal_id.pkl', 'wb') as f:
    pkl.dump(token2traversal_id, f)

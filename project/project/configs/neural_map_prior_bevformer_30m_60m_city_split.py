"""
BEVformer map baseline with resnet101 backbone and neural map prior
"""

_base_ = [
    './default_runtime.py'
]

# TODO 23863MB for single-GPU training
plugin = True
plugin_dir = 'project/neural_map_prior/'
find_unused_parameters = True
# sync_bn = True


# data root and data info path for dataset
version = 'trainval'  # 'mini' or 'trainval'
data_root = '/public/MARS/datasets/nuScenes/'
data_info_path = '/public/MARS/datasets/nuScenes/'

# input_modality
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# define img_norm_cfg
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False
)

map_grid_conf = {
    'xbound': [-30.0, 30.0, 0.15],
    'ybound': [-15.0, 15.0, 0.15],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0],
}

map_bev_attrs = dict(
    real_h_=30,
    real_w_=60,
    bev_h_=100,
    bev_w_=200,
)

# Default class names.  Never used
class_names = [
    'ped_crossing',  # 0
    'road_divider', 'lane_divider',  # 1
    'contours',  # 3
    'others',  # -1
]

class2label = {
    'ped_crossing': 0,
    'divider': 1,
    'contours': 2,
    'others': -1,
}
num_class = max(list(class2label.values())) + 1

# if input_modality['use_lidar']:
#     lidar_dim = 128
#     use_lidar = True
# else:
#     lidar_dim = 0
#     use_lidar = False

angle_class = 36
direction_pred = True
lidar_dim = 128
rasterized_cfg = dict(
    raw_dataset_cfg=dict(
        version=version,
        data_root=data_root,
    ),
    data_aug_conf={
        'line_width': 5,
        'direction_pred': direction_pred,
        'angle_class': angle_class
    },
    grid_conf=map_grid_conf,
)

bevformer_dim = 256
bevformer_cfg = dict(
    type='BEVformer',
    _dim_=bevformer_dim,
    _pos_dim_=bevformer_dim // 2,
    _ffn_dim_=bevformer_dim * 2,
    _num_levels_=4,
    queue_length=3,  # each sequence contains `queue_length` frames.
    point_cloud_range=[-15.0, -30.0, -5.0, 15.0, 30.0, 3.0],
)

map_attribute = {
    'root_dir': '/localdata_ssd/map_slices/raster_global_map',
    'type': 'rasterized',
    'prefix': 'map_large_reso_gru_cpu',
    'tile_param': {
        'data_type': 'float32',
        'embed_dims': 256,
        'num_traversals': 1, },
    'batch_size': 8,
    'single_gpu': False,
    'global_map_tile_size': [4, 4],
    'global_map_raster_size': [0.30, 0.30],
    'nusc_city_split': True,
}
sort_train_infos = True
sort_val_infos = True
sort_batch_size = map_attribute['batch_size']

model = dict(
    type='NeuralMapPrior',
    map_bev_attrs=map_bev_attrs,
    dist_cfg=False,
    map_attribute=map_attribute,
    open_nmp=True,
    use_centerness=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(
            type='BN2d',
            requires_grad=False
        ),
        norm_eval=True,
        style='caffe',
        dcn=dict(
            type='DCNv2',
            deform_groups=1,
            fallback_on_stride=False
        ),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=bevformer_dim,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=bevformer_cfg['_num_levels_'],
        relu_before_extra_convs=True),
    positional_encoding_prior=dict(
        type='LearnedPositionalEncoding',
        num_feats=bevformer_cfg['_pos_dim_'],
        row_num_embed=map_bev_attrs['bev_h_'],
        col_num_embed=map_bev_attrs['bev_w_'],
    ),
    positional_encoding_cur=dict(
        type='LearnedPositionalEncoding',
        num_feats=bevformer_cfg['_pos_dim_'],
        row_num_embed=map_bev_attrs['bev_h_'],
        col_num_embed=map_bev_attrs['bev_w_'],
    ),
    view_transformation_cfg=dict(
        type='BEVFormer',
        embed_dims=bevformer_dim,
        num_feature_levels=4,
        num_cams=6,
        use_cams_embeds=True,
        map_bev_attrs=map_bev_attrs,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bevformer_cfg['_pos_dim_'],
            row_num_embed=map_bev_attrs['bev_h_'],
            col_num_embed=map_bev_attrs['bev_w_'],
        ),
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=6,
            pc_range=bevformer_cfg['point_cloud_range'],
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type='MapPriorLayer',
                attn_cfgs=[
                    dict(type='TemporalSelfAttention',
                         embed_dims=bevformer_dim,
                         num_levels=1),
                    dict(type='SpatialCrossAttention',
                         pc_range=bevformer_cfg['point_cloud_range'],
                         deformable_attention=
                         dict(type='MSDeformableAttention3D',
                              embed_dims=bevformer_dim,
                              num_points=16,
                              num_levels=bevformer_cfg['_num_levels_']),
                         embed_dims=bevformer_dim,
                         ),
                    dict(
                        type='WindowCrossAttention',
                        num_bev_win_h=10,
                        num_bev_win_w=20,
                        bev_h=map_bev_attrs['bev_h_'],
                        bev_w=map_bev_attrs['bev_w_'],
                        embed_dims=bevformer_dim,
                    )
                ],
                feedforward_channels=bevformer_cfg['_ffn_dim_'],
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm',
                                 'cross_attn', 'win_cross_attn', 'norm',
                                 'ffn', 'norm')
            )
        ),
    ),
    head_cfg=dict(
        type='BevEncode',
        inC=256 + lidar_dim if input_modality['use_lidar'] else 256,
        outC=4,
        instance_seg=True,
        embedded_dim=16,
        direction_pred=True,
        direction_dim=36 + 1,
    ),
    loss_cfg=dict(
        type='HdmapnetLoss',
        angle_class=angle_class,
        direction_pred=direction_pred,
        loss_seg=dict(
            type='CrossEntropyLoss',
            class_weight=[1.0, 15.0, 15.0, 15.0],
            loss_weight=2.0),
        loss_embed=dict(
            type='DiscriminativeLoss',
            embed_dim=16,
            delta_v=0.5,
            delta_d=3.0,
            loss_weight_cfg=dict(
                var=1.,
                reg=1.,
                dist=1.,
            )
        ),
        loss_dir=dict(
            type='DirectionLoss',
            loss_weight=0.2,
        ),
    ),
)

train_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles'),
    dict(type='ResizeCameraImage',
         fW=1600, fH=900,
         W=1600, H=900, ),
    dict(
        type='VectorizeLocalMap',
        data_root=data_root,
        patch_size=(30, 60),
        sample_dist=0.7,
        num_samples=150,
        sample_pts=True,
        max_len=120,
        padding=False,
        normalize=True,
        fixed_num={
            'ped_crossing': 100,
            'divider': 100,
            'contours': 100,
            'others': -1, },
        class2label=class2label,
    ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='Pad3D', size_divisor=32),
    dict(type='FormatBundleMap', with_gt=False),
    dict(type='Collect3D',
         keys=['img', 'rasterized_gt', 'vectors'],
         meta_keys=('img_filenames', 'ori_shape',
                    'img_shape', 'lidar2img',
                    'pad_shape', 'scale_factor',
                    'flip', 'cam_intrinsics',
                    'img_norm_cfg', 'sample_idx',
                    'cam2ego_rotations', 'cam2ego_translations',
                    'ego2global_translation', 'ego2global_rotation',
                    'ego2img', 'yaw_pitch_roll',
                    'post_trans', 'post_rots',))
]

test_pipeline = train_pipeline

# output evalution format
eval_cfg = dict(
    raster=True,
    patch_size=(30, 60),
    canvas_size=(200, 400),
    evaluation_cfg=dict(
        version='v1.0-trainval',
        result_path='./',
        dataroot=data_root,
        eval_set='val',
        num_class=num_class,
        max_channel=3,
        bsz=64,
        CD_threshold=10,
        thickness=5,
        xbound=map_grid_conf['xbound'],
        ybound=map_grid_conf['ybound'],
        class_name=class_names,
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=10,
    train=dict(
        type='nuScenesMapDataset',
        data_root=data_root,
        ann_file=data_info_path + 'nuScences_map_trainval_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        eval_cfg=eval_cfg,
        rasterized_cfg=rasterized_cfg,
        sort_train_infos=sort_train_infos,
        sort_val_infos=sort_val_infos,
        sort_batch_size=sort_batch_size,
        gm_grid_size=map_attribute['global_map_tile_size'],
        nusc_city_split=True,
    ),
    val=dict(
        type='nuScenesMapDataset',
        data_root=data_root,
        ann_file=data_info_path + 'nuScences_map_trainval_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        eval_cfg=eval_cfg,
        rasterized_cfg=rasterized_cfg,
        sort_train_infos=sort_train_infos,
        sort_val_infos=sort_val_infos,
        sort_batch_size=sort_batch_size,
        gm_grid_size=map_attribute['global_map_tile_size'],
        nusc_city_split=True,
    ),
    test=dict(
        type='nuScenesMapDataset',
        data_root=data_root,
        ann_file=data_info_path + 'nuScences_map_trainval_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        eval_cfg=eval_cfg,
        rasterized_cfg=rasterized_cfg,
        samples_per_gpu=1,
        sort_train_infos=sort_train_infos,
        sort_val_infos=sort_val_infos,
        sort_batch_size=sort_batch_size,
        gm_grid_size=map_attribute['global_map_tile_size'],
        nusc_city_split=True,
    ),
)
custom_hooks = [dict(type='SetEpochInfoHook', priority='VERY_HIGH', interval=1)]

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'img_backbone': dict(lr_mult=0.1)}))

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# kwargs for dataset evaluation
evaluation = dict(interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=24)

load_from = '/oldhome/xiongx/repository/neural_map_prior_code/work_dirs/bevformer_30m_60m_city_split/epoch_24.pth'

file_client_args = dict(backend='disk')

# Getting started with Neural Map Prior

## Add custom data_sampler

In our setup, we need to manually replace mmdet `mmdet/apis/train.py` with ours `tools/mmdet_train.py` to add a new
custom data_sampler.
If you know how to add a new data_sampler through config, please let us know through a pull request.

## Training

Neural map prior incorporates BEVFormer with ResNet101 backbone as the baseline, using the weight initialization from
the ckpts/r101_dcn_fcos3d_pretrain.pth files. It can be downloaded
from [here](https://drive.google.com/file/d/1bkiDxA97XvhnRQnGB44ol3xwhVjGcffu/view?usp=drive_link).

### Multi-GPU training

To train neural_map_prior with 8 GPUs, run:

```bash
bash tools/dist_train.sh $CONFIG 8
```

For example, if you want to train baseline and neural_map_prior with 8 GPUs on nuScenes dataset, run:

```bash
bash tools/dist_train.sh project/configs/bevformer_30m_60m.py 8
bash tools/dist_train.sh project/configs/neural_map_prior_bevformer_30m_60m.py 8
```

### Single GPU training

```bash
python tools/train.py $CONFIG
```

## Evaluation

### Multi-GPU evaluation

To evaluate neural_map_prior with 8 GPU, run:

```bash
bash tools/dist_test.sh $YOUR_CONFIG $YOUR_CKPT 8 --eval iou
```

For example, if you want to evaluate the baseline and neural_map_prior with 8 GPUs on nuScenes dataset, run:

```bash
bash tools/dist_test.sh project/configs/bevformer_30m_60m.py $YOUR_CKPT 8 --eval iou
bash tools/dist_test.sh project/configs/neural_map_prior_bevformer_30m_60m.py $YOUR_CKPT 8 --eval iou
```

### Single GPU evaluation

```bash
python tools/test.py $YOUR_CONFIG $YOUR_CKPT --eval iou
```

## Visualization

To visualize the predictions, run:

```bash
python project/neural_map_prior/map_tiles/lane_render.py
```

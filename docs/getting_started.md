# Getting started with Neural Map Prior

## Before training

### step 1. computation requirements

Make sure that you have over 50GB of memory available. It is required to create map tiles for the four cities in nuScene
in the training and testing process.

### step 2. add custom data_sampler

In our setup, we need to manually replace mmdet `mmdet/apis/train.py` with ours `tools/mmdet_train.py` to add a new
custom data_sampler.
If you know how to add a new data_sampler through config, please let us know through a pull request.

## Training

Neural map prior incorporate two stages to get the desired results: the first stage is to obtain baseline results, and
the second stage is to fine-tune with fusion modules and global memory.

### Stage 1: Obtain the baseline

We begin by training the BEV feature extraction module to achieve stable weight initialization for the following stage

1. ~23GB GPU Memory, ~1.5 days for 24 epochs on 8 3090 GPUs
2. We found the results of baseline training at epoch 24 or
   epoch 100 to be similar, so 24 epochs is enough for baseline.

#### Multi-GPU training

To train neural_map_prior with 8 GPUs, run:

```bash
bash tools/dist_train.sh $CONFIG 8
```

For example, if you want to train baseline and neural_map_prior with 8 GPUs on nuScenes dataset, run:

```bash
bash tools/dist_train.sh project/configs/bevformer_30m_60m.py 8
```

Note: Neural map prior incorporates BEVFormer with ResNet101 backbone as the baseline, using the weight initialization
from
the ckpts/r101_dcn_fcos3d_pretrain.pth files. It can be downloaded
from [here](https://drive.google.com/file/d/1bkiDxA97XvhnRQnGB44ol3xwhVjGcffu/view?usp=sharing).

You can create the ckpts directory in this repository root and move downloaded checkpoints to ckpts.

```bash
mkdir ckpts && cd ckpts
```

### Stage 2: Finetune with NMP

Based on the last epoch obtained by the baseline training (epoch_24 in our setting), which is set in `load_from` in
config, fine-tune another 24 epochs to achieve the NMP advantage brought to the table. In the time of fine-tuning, we
freeze the training of the backbone and neck.
(~ 12GB GPU Memory, ~1 days for 24 epochs on 8 3090 GPUs)

```bash
bash tools/dist_train.sh project/configs/neural_map_prior_bevformer_30m_60m.py 8
```

#### Single GPU training

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
```

(To evaluate neural_map_prior, Be careful to set the appropriate `data_sample.py` that needs to be used in NMP.)

```bash
bash tools/dist_test.sh project/configs/neural_map_prior_bevformer_30m_60m.py $YOUR_CKPT 8 --eval iou
```

### Single GPU evaluation

```bash
python tools/test.py $YOUR_CONFIG $YOUR_CKPT --eval iou
```

## Visualization

To visualize the predictions, you can first try with nmp_results, which you can download
from [here](https://drive.google.com/file/d/1vcajqEPfIJ_Vb4jrG4umoKY_qp3ZrsCE/view?usp=sharing), and run:

```bash
python project/neural_map_prior/map_tiles/lane_render.py
```

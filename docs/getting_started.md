# Getting started with Neural Map Prior

## Evaluation Test

Please ensure that the environment and the nuScenes dataset are ready. You can test it by assessing the pre-trained
first-stage baseline model as shown below:

```bash
cd neural_map_prior
mkdir ckpts && cd ckpts
```

```bash
cd ..
./tools/dist_test.sh ./project/configs/bevformer_30m_60m.py ./ckpts/bevformer_epoch_24.pth 8 --eval iou
```

If everything is done correctly, the output should be:
(if you test with a different number of GPUs than 8, the results may differ slightly.)

```markdown
Divider | Crossing | Boundary | All(mIoU)

49.20 | 28.67 | 50.43 | 42.76
```

## Before Neural Map Prior Training

### step 1. computation requirements

Make sure that you have over 50GB of memory available. It is required to create map tiles for the four cities in nuScene
in the training and testing process.

### step 2. add custom data_sampler

In our setup, to add a new custom data_sampler, we need to manually copy the `tools/mmdet_train.py` file to the path of
the installed package mmdet located in `mmdet/apis/*`. Then, we rename the copied mmdet_train.py file to train.py.
Additionally, we need to ensure the project root is added to the PYTHONPATH so that it can
reference `tools/data_sampler.py`.
If you know how to add a new data_sampler through config, please let us know through a pull request.

Notes: Whether for training or evaluation, remember to change the root directory addresses of the data_infos/city_infos
in these two areas.

1. `project/configs/bevformer_30m_60m.py` or `project/configs/neural_map_prior_bevformer_30m_60m.py`

   ```python
   data_root = 'path/to/data/nuscenes/'
   data_info_path = 'path/to/data/nuscenes/'
   ```

2. Within the two functions in this file `project/neural_map_prior/map_tiles/lane_render.py`
   `load_nusc_data_infos` and `load_nusc_data_cities`:

   ```python
   # load_nusc_data_infos 
   root = 'path/to/data/nuscenes'
   # load_nusc_data_cities
   root = 'path/to/data/nuscenes_info'
   ```

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

For example, if you want to train baseline with 8 GPUs on nuScenes dataset, run:

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

Building upon the last epoch achieved during baseline training (epoch_24 in our setting), which is specified in the
`load_from` configuration, we perform fine-tuning for an additional 24 epochs to leverage the advantages of NMP. During
the fine-tuning process, we keep the training of the backbone and neck frozen.
(~ 13GB GPU Memory, ~ 1 days for 24 epochs on 8 3090 GPUs)

For example, if you want to train neural map prior with 8 GPUs on nuScenes dataset, run:

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
bash tools/dist_test.sh project/configs/bevformer_30m_60m.py ./ckpts/bevformer_epoch_24.pth 8 --eval iou
```

(To evaluate neural_map_prior, Be careful to set the appropriate `data_sample.py` that needs to be used in NMP.)

```bash
bash tools/dist_test.sh project/configs/neural_map_prior_bevformer_30m_60m.py ./ckpts/nmp_epoch_24.pth 8 --eval iou
```

### Single GPU evaluation

```bash
python tools/test.py $YOUR_CONFIG $YOUR_CKPT --eval iou
```

## Visualization

To visualize the predictions, you can first try with nmp_results, which you can download
from [here](https://drive.google.com/file/d/1vcajqEPfIJ_Vb4jrG4umoKY_qp3ZrsCE/view?usp=sharing), upzip and run:

```bash
python project/neural_map_prior/map_tiles/lane_render.py --result_root 'nmp_results'
```

<div align="center">

# InsMOS: Instance-Aware Moving Object Segmentation in LiDAR Data

 [![](https://img.shields.io/badge/Page-InsMOS-d6ccc2?style=flat-square&labelColor=495057&logoColor=white)](https://neng-wang.github.io/InsMOS/) [![](https://img.shields.io/badge/Paper-IEEE-023e8a?style=flat-square&labelColor=495057&logoColor=white)](https://ieeexplore.ieee.org/document/10342277) [![](https://img.shields.io/badge/Paper-arXiv-d62828?style=flat-square&labelColor=495057&logoColor=white)](https://arxiv.org/abs/2303.03909)

</div>



This repository contains the implementation of our paper:

> **InsMOS: Instance-Aware Moving Object Segmentation in LiDAR Data**\
> [Neng Wang](https://github.com/neng-wang),  [Chenghao Shi](https://github.com/chenghao-shi),  Ruibin Guo,  Huimin Lu,  Zhiqiang Zheng,  [Xieyuanli Chen](https://github.com/Chen-Xieyuanli)   

<div align=center>
<img src="./docs/InsMOS.gif"> 
</div>

- *Our instance-aware moving object segmentation on the SemanticKITTI sequence 08 and 20, 21.*

- *Red points indicate predicted moving points, cyan indicate predicted static instance points and gray points are static background.*

- *Green bounding boxes represent cars, blue bounding boxes represent pedestrians, and yellow bounding boxes represent cyclists.*

## News

- [2023-8-12] Code released!
- [2023-6-22] Our work is accepted for IROS2023 :clap:

  
## Citation

If you use our code in your work, please star our repo and cite our paper.

```bibtex
@inproceedings{wang2023iros,
	title={{InsMOS: Instance-Aware Moving Object Segmentation in LiDAR Data}},
	author={Wang, Neng and Shi, Chenghao and Guo, Ruibin and Lu, Huimin and Zheng, Zhiqiang and Chen, Xieyuanli},
	booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
        pages={7598-7605},
	year={2023}
}
```

## Data

1、SemanticKITTI: Download SemanticKITTI dataset from the official [website](http://semantic-kitti.org/). 

2、KITTI-road Dataset: Download the KITTI-road Velodyne point clouds from the official [website](https://www.cvlibs.net/datasets/kitti/raw_data.php?type=road) and MOS label from [MotionSeg3D](https://github.com/haomo-ai/MotionSeg3D).

3、Instance label:   Download  the box labels from [ondrive](https://1drv.ms/f/s!Ak6KrcxOqwZfkABaeJYYLb7ZT7Fg?e=zguXiK) or [BaiduDisk,code:59t7](https://pan.baidu.com/s/1TVBED6KZmEsJI6R_xjdLRQ?pwd=59t7), and please refer to [boundingbox_label_readme](./dataloader/boundingbox_label_readme.md) about more details of instance label .

<details>
    <summary><strong>Data structure</strong></summary>

```
└── sequences
  ├── 00/           
  │   ├── velodyne/	
  |   |	├── 000000.bin
  |   |	├── 000001.bin
  |   |	└── ...
  │   ├── labels/ 
  |   | ├── 000000.label
  |   | ├── 000001.label
  |   | └── ...
  |   ├── boundingbox_label
  |   | ├── 000000.npy
  |   |	├── 000001.npy
  |   |	└── ...
  |   ├── calib.txt
  |   ├── poses.txt
  |   └── times.txt
  ├── 01/ # 00-10 for training
  ├── 08/ # for validation
  ├── 11-21/ # 11-21 for testing
  # kitti-road
  ├── 30 31 32 33 34 40 # for training
  └── 35 36 37 38 39 41 # for testing
```

</details> 

## Installation

#### 1. Dependencies

 **system dependencies:**

```bash
ubuntu20.04, CUDA 11.3, cuDNN 8.2.1, 
```

**python dependencies:**

```bash
python 3.7
```

#### 2. Set up conda environment

```bash
conda create --name insmos python=3.7
conda activate insmos
pip install -r requirements.txt

# insltall pytorch with cuda11.3, avoid using "pip install torch"
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# ensure numpy==1.18.1
pip uninstall numpy
pip install numpy==1.18.1
```

**Install MinkowskiEngine :**

```bash
cd ~
mkdir ThirdParty
sudo apt-get install libopenblas-dev
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
conda activate insmos
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

#### 3. Install InsMOS

```bash
# clone code
git clone https://github.com/nubot-nudt/InsMOS.git
cd InsMOS

# activate conda
conda activate insmos

# install
python setup.py develop
```

## Inference

Run the following command to evaluate the model in SemanticKITTI validation dataset or test dataset. At this moment, a “preb_out” folder will be generated, which contains  “bbox_preb”, “confidence”, and “mos_preb” for storing the predicted  bounding boxes, confidence scores for moving points, and labels for  moving points, respectively.

We public the model was trained on the SemanticKITTI dataset (N_10_t_0.1_odom.ckpt) and the other model was trained on the Semantic-KITTI and KITTI-road dataset (N_10_t_0.1_odom_road.ckpt). You can download from  [ondrive](https://1drv.ms/f/s!Ak6KrcxOqwZfkABaeJYYLb7ZT7Fg?e=zguXiK) or [BaiduDisk,code:59t7](https://pan.baidu.com/s/1TVBED6KZmEsJI6R_xjdLRQ?pwd=59t7), and then put the model in "ckpt" folder.

```bash
cd InsMOS
python scripts/predict_mos.py --cfg_file config/config.yaml --data_path /path/to/kitti/sequences --ckpt ./ckpt/N_10_t_0.1_odom.ckpt --split valid
```

### Evaluate

We use the [semantickitti-api](https://github.com/PRBonn/semantic-kitti-api) to evaluate the MOS IOU.

```bash
cd semantic-kitti-api
python evaluate_mos.py --dataset /path/to/kitti --predictions ./preb_out/InsMOS/mos_preb --split valid
```

### Refine

Run the following command to refine the network predictions.

```bash
python scripts/refine.py --data_path /path/to/kitti/sequences --split valid
```

### Re-evaluate the refinement

Re-evaluate the results of refinement.

```bash
python evaluate_mos.py --dataset /path/to/kitti --predictions ./preb_out_refine/mos_preb --split valid 
```

### Visual

Run the following command to visualize the results of moving object segmentation and instance prediction.

Press key  `n`  to show next frame.

Press key  `b`  to show last frame.

Press key  `q`  to quit display.

```bash
cd visual
python vis_mos_bbox.py
```

### Train

You can set  single gpu or multi gpu for training  in [train.py](./scripts/train.py). We set batch size to 4 for each gpu. During the training process, there may be an increase in GPU memory  consumption, so it is advisable not to set the batch size too large  initially. We test 4-6 is fine on  3090 GPU. 

```bash
export DATA=/path/to/kitti/sequences
python scripts/train.py
```

If the training process is interrupted unexpectedly, you can resume the training using the following command. 

```bash
python scripts/train.py --weights ./logs/InsMOS/version_x/checkpoints/last.ckpt --checkpoint ./logs/InsMOS/version_x/checkpoints/last.ckpt
```

## Contact

Any question or suggestions are welcome!

Neng Wang: nwang@nudt.edu.cn and Xieyuanli Chen: [website](https://github.com/Chen-Xieyuanli)

## License

This project is free software made available under the MIT License. For details see the LICENSE file.


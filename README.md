<center>

# [Coarse-to-Fine Amodal Segmentation with Shape Prior (C2F-Seg)](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Coarse-to-Fine_Amodal_Segmentation_with_Shape_Prior_ICCV_2023_paper.pdf)

[Jianxiong Gao](https://jianxgao.github.io/), [Xuelin Qian†](https://naiq.github.io/), [Yikai Wang](https://yikai-wang.github.io/), [Tianjun Xiao†](https://tianjunxiao.com/), [Tong He](https://hetong007.github.io/), [Zheng Zhang](https://www.amazon.science/author/zheng-zhang), [Yanwei Fu](https://yanweifu.github.io/)

[![ArXiv](https://img.shields.io/badge/ArXiv-2308.16825-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2308.16825)
[![HomePage](https://img.shields.io/badge/HomePage-Visit-blue.svg?logo=homeadvisor&logoColor=f5f5f5)](https://jianxgao.github.io/C2F-Seg/)
[![Dataset](https://img.shields.io/badge/Dataset-MOViD_Amodal-F07B3F.svg)](https://data.dgl.ai/dataset/C2F-Seg/MOViD_A.tar)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://github.com/amazon-science/c2f-seg/blob/main/LICENSE) 
[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Famazon-science%2Fc2f-seg&count_bg=%2352D3D8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors&edge_flat=false)](https://hits.seeyoufarm.com)

</center>

# Introduction

<img src='./imgs/C2F-Seg.jpg' width="100%">


# Environment Setup

```bash
git clone https://github.com/amazon-science/c2f-seg.git
cd c2f-seg
conda create --name C2F_Seg --file requirements.txt
```

# Dataset and checkpoints

| Dataset      |   $\text{mIoU}_{full}$ |    $\text{mIoU}_{occ}$      |  VQ Model        | C2F-Seg  |
| :---         |   :---:                |   :---:                     |    :---:   |   :---:          | 
| [KINS](https://data.dgl.ai/dataset/C2F-Seg/KINS.tar)         |     82.22             |   53.60        |  [weight](https://data.dgl.ai/dataset/C2F-Seg/vqgan_KINS.pth), [config](https://data.dgl.ai/dataset/C2F-Seg/vqgan_KINS.yml)              | [weight](https://data.dgl.ai/dataset/C2F-Seg/c2f_seg_KINS.pth), [config](https://data.dgl.ai/dataset/C2F-Seg/c2f_seg_KINS.yml)    |
| [COCOA](https://data.dgl.ai/dataset/C2F-Seg/COCOA.tar)        |     80.28              |    27.71       |  [weight](https://data.dgl.ai/dataset/C2F-Seg/vqgan_COCOA.pth), [config](https://data.dgl.ai/dataset/C2F-Seg/vqgan_COCOA.yml)            | [weight](https://data.dgl.ai/dataset/C2F-Seg/c2f_seg_COCOA.pth), [config](https://data.dgl.ai/dataset/C2F-Seg/c2f_seg_COCOA.yml)    |
| [MOViD-Amodal](https://data.dgl.ai/dataset/C2F-Seg/MOViD_A.tar) |     71.67              |    36.13       |  [weight](https://data.dgl.ai/dataset/C2F-Seg/vqgan_MOViD_A.pth), [config](https://data.dgl.ai/dataset/C2F-Seg/vqgan_MOViD_A.yml)            | [weight](https://data.dgl.ai/dataset/C2F-Seg/c2f_seg_MOViD_A.pth), [config](https://data.dgl.ai/dataset/C2F-Seg/c2f_seg_MOViD_A.yml)    |

**Please use the following commands to prepare the dataset and checkpoints:**
```bash
# Example with KINS dataset
bash download.sh KINS
wget https://data.dgl.ai/dataset/C2F-Seg/KINS.tar
tar -xvf KINS.tar
# Important: Update the root_path in config files!
```

### Exmaple of Proposed MOViD-Amodal:
<img src="./imgs/example.gif" width="100%">

# Running Experiments

### Evaluate model

```bash
# KINS
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
test_c2f_seg.py --dataset KINS --batch 1 --data_type image --vq_path KINS_vqgan --path KINS_c2f_seg

# MOViD-Amodal
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
test_c2f_seg.py --dataset MOViD_A --batch 1 --data_type video --vq_path MOViD_A_vqgan --path MOViD_A_c2f_seg 
```

### Train VQ model

```bash
# KINS
CUDA_VISIBLE_DEVICES=0 python train_vq.py --dataset KINS --path KINS_vqgan --check_point_path ../check_points
# MOViD-Amodal
CUDA_VISIBLE_DEVICES=0 python train_vq.py --dataset MOViD_A --path MOViD_A_vqgan --check_point_path ../check_points
```

### Train C2F-Seg

```bash
# KINS
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
train_c2f_seg.py --dataset KINS --batch 16 --data_type image --vq_path KINS_vqgan --path KINS_c2f_seg
# MOViD-Amodal
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
train_c2f_seg.py --dataset MOViD_A --batch 1 --data_type video --vq_path MOViD_A_vqgan --path MOViD_A_c2f_seg 
```


# Citation
If you find our paper useful for your research and applications, please cite using this BibTeX:
```
@inproceedings{gao2023coarse,
  title={Coarse-to-Fine Amodal Segmentation with Shape Prior},
  author={Gao, Jianxiong and Qian, Xuelin and Wang, Yikai and Xiao, Tianjun and He, Tong and Zhang, Zheng and Fu, Yanwei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1262--1271},
  year={2023}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


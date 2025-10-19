# RCT

This repo is the official project repository of the paper ***Rapid Structure-agnostic Single Crystal X-Ray Diffraction Indexing via Reflection Cloud Learning*** [ [arxiv](https://) ] [ [RC-400K](https://) ]

Authors: 

## Installation

我们建议对数据模拟部分和模型训练部分使用分别的环境避免潜在环境冲突问题，建议使用Nvidia BlackWell架构GPU，如GeForce RTX 50 Series。。如果只是复现论文结果，你可以跳过数据模拟步骤直接使用RC-400K数据集。数据集将在论文建刊后开源。

Prerequisites [[uv](https://astral.sh/uv/)] [[mamba](https://mamba.readthedocs.io/)]

### Simulator Environment

We build our simulator based on [DATAD](https://datad.netlify.app/). Please first apply for datad.whl file at [here](https://datad.netlify.app/install).

```bash

```

### RCT Environment

```bash
mamba create -n rct python=3.12
mamba activate rct

mamba install -c nvidia/label/cuda-12.8.0 cuda-toolkit -y
mamba install cudnn gcc gxx -y

uv pip install ninja
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install h5py pyyaml
uv pip install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm matplotlib
uv pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install spconv_cu126-2.3.8-cp312-cp312-manylinux_2_28_x86_64.whl
uv pip install flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl

cd libs/pointops
python setup.py install
cd ../..

```

## Run

<details>
<summary> Produce simulated reflection cloud dataset</summary>

</details>

## Links

RCT is based on [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3), 并使用了[pointcept](https://github.com/Pointcept/Pointcept)的代码实现。

**Point Transformer V3: Simpler, Faster, Stronger**
Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2024.
[[paper](https://arxiv.org/pdf/2312.10035)] [[code](https://github.com/Pointcept/PointTransformerV3)] [[pointcept](https://github.com/Pointcept/Pointcept)]

## Bibtex

If you find our work helpful, please consider citing the following BibTeX entry.

```bibtex

```
# RCT

This repo is the official project repository of the paper ***Rapid Structure-agnostic Single Crystal X-Ray Diffraction Indexing via Reflection Cloud Learning*** [ [arxiv](https://) ] [ [RC-400K](https://) ]

Authors: 

## Environment Setup

我们建议对数据模拟部分和模型训练部分使用分别的环境，避免潜在环境冲突问题。对于论文结果复现，你可以跳过数据模拟步骤，直接使用RC-400K数据集进行训练和测试。数据集将在论文建刊后开源。
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
<summary> Produce simulate reflection cloud dataset</summary>

</details>

## Links

We use [DATAD](https://datad.netlify.app/) as our XRD simulator.
RCT is based on [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3), 并从[pointcept](https://github.com/Pointcept/Pointcept)中借鉴了相关实现。


## Bibtex

If you find our work helpful, please consider citing the following BibTeX entry.

```bibtex

```
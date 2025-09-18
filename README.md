# XRDT

XRDT is a deep learning framework for X-ray diffraction analysis and simulation.

## Environment Setup

This project requires two separate environments:

### 1. Simulator Environment
*For generating simulated datasets*

*Configuration details to be added*

### 2. XRD Training Environment
*For model training*

#### Prerequisites
- [uv](https://astral.sh/uv/) package manager
- [mamba](https://mamba.readthedocs.io/) package manager

#### Installation Steps

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate conda environment
mamba create -n xrd_tf python=3.12
mamba activate xrd_tf

# Install CUDA toolkit and dependencies
mamba install -c nvidia/label/cuda-12.8.0 cuda-toolkit -y
mamba install cudnn gcc gxx -y

# Install build tools
uv pip install ninja

# Install PyTorch
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
uv pip install h5py pyyaml
uv pip install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm

# Install PyTorch Geometric
uv pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Install specialized packages
uv pip install spconv_cu126-2.3.8-cp312-cp312-manylinux_2_28_x86_64.whl
uv pip install flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl

# Install point operations library
cd libs/pointops
python setup.py install
cd ../..

# Install visualization tools
uv pip install matplotlib
```

## Usage

*Usage instructions to be added*

## License

*License information to be added*

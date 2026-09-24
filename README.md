# SWINGOT

## Overview

SWINGOT integrates paired spatial RNA–ATAC or RNA–ADT data and produces a fused spot-level representation for clustering and visualization.

## Requirements and Installation

[![Python 3.8.20](https://img.shields.io/badge/Python-3.8.20-blue)](https://www.python.org/)
[![PyTorch 2.1.1 + CUDA 12.1](https://img.shields.io/badge/PyTorch-2.1.1%2Bcu121-orange)](https://pytorch.org/)
[![Scanpy 1.9.8](https://img.shields.io/badge/Scanpy-1.9.8-green)](https://scanpy.readthedocs.io/)
[![BEDTools 2.31.1](https://img.shields.io/badge/BEDTools-2.31.1-purple)](https://bedtools.readthedocs.io/)

The commands below follow the exported SWINGOT Linux server environment. Run them on a Linux server with a CUDA 12.1-compatible NVIDIA driver.

```bash
git clone https://github.com/zhangdaoliang/SWINGOT.git
cd SWINGOT

conda create -n SWINGOT -c bioconda -c conda-forge -c defaults python=3.8.20 pip=24.2 bedtools=2.31.1
conda activate SWINGOT

python -m pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Check the installation from the repository root:

```bash
bedtools --version
python -c "import pybedtools, scanpy, torch, swingot; from swingot import SWINGOT; print(SWINGOT.__name__, torch.__version__, scanpy.__version__)"
```

## Datasets

Mouse embryo data are available from the [Allen Developing Mouse Brain Atlas](https://developingmouse.brain-map.org/). Human lymph node data are available from GEO ([GSE263617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263617)).

## Tutorials

Run a script from the repository root after preparing its input data:

- E13: `python Benchmark/SWINGOT/runSWINGOT2.py --seed 0 --tag seed0`
- E15: `python Benchmark/SWINGOT/runSWINGOT1.py --seed 0 --tag seed0`
- E18: `python Benchmark/SWINGOT/runSWINGOT3.py --seed 0 --tag seed0`
- Human lymph node: `python Benchmark/SWINGOT/runSWINGOT4.py --seed 0 --tag seed0`
- Simulation3: `python Benchmark/SWINGOT/runSWINGOT5.py --seed 0 --tag seed0`

The original SWITCH comparison is under `Benchmark/others/integration/` with the other methods. Run `python Benchmark/others/integration/runSWITCH.py --dataset E15 --seed 0 --data-root /path/to/data`.

## Contact

For questions, please [open a GitHub issue](https://github.com/zhangdaoliang/SWINGOT/issues) or contact us at zhangdaoliang@fudan.edu.cn wangjikuo@foxmail.com for problems about the packages..

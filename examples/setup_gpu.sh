#!/usr/bin/env bash
# Minimal GPU setup for the Dayhoff quickstart notebook (full model generation + scoring).
# Requires a CUDA 12.8 toolkit (nvcc) to build the kernels. Creates a conda env,
# installs everything, and registers a Jupyter kernel.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

conda create -y -n dayhoff-gpu python=3.12
conda run -n dayhoff-gpu pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n dayhoff-gpu pip install -e .            # also pulls transformers/datasets/huggingface_hub
conda run -n dayhoff-gpu pip install --no-build-isolation mamba-ssm==2.3.1 flash-attn==2.8.3 causal-conv1d==1.6.0
conda run -n dayhoff-gpu pip install ipykernel ipywidgets
conda run -n dayhoff-gpu python -m ipykernel install --user --name dayhoff-gpu --display-name "Python (dayhoff-gpu)"

echo "Done. Open examples/dayhoff_quickstart.ipynb and select the 'Python (dayhoff-gpu)' kernel."

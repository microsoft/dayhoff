#!/usr/bin/env bash
# Minimal CPU setup for the Dayhoff quickstart notebook (imports + dataset loading).
# Creates a conda env, installs the package, and registers a Jupyter kernel.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

conda create -y -n dayhoff-cpu python=3.12
conda run -n dayhoff-cpu pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n dayhoff-cpu pip install -e .            # also pulls transformers/datasets/huggingface_hub
conda run -n dayhoff-cpu pip install ipykernel ipywidgets
conda run -n dayhoff-cpu python -m ipykernel install --user --name dayhoff-cpu --display-name "Python (dayhoff-cpu)"

echo "Done. Open examples/dayhoff_quickstart.ipynb and select the 'Python (dayhoff-cpu)' kernel."

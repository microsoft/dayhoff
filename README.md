# Project
## Description

## Installation
**Requirements**: 
* PyTorch: 2.2 and above
* CUDA 12.0 and above
* Optionally install Flash Attention following installation instructions here: https://github.com/Dao-AILab/flash-attention

To download our code, we recommend creating a clean conda environment with Python 3.10.16

```
conda create --name dayhoff python=3.10.16
```

In that new environment, install PyTorch, mamba-ssm, and causal-conv1d. Then install Dayhoff. Optionally, install Flash Attention 1 or 2.

```
pip install dayhoff # For bleeding edge: pip install git+https://github.com/microsoft/dayhoff.git
```
### Mamba-ssm and causal-conv1d recommendations

It is sometimes challenenging to properly install these packages just using pip. The following two errors are common when simply using pip install:
* packages installed correctly, but when loading models you get "ValueError: Fast Mamba kernels are not available. Make sure to they are installed and that the mamba module is on a CUDA device."
* Package instalation of causal-conv1d or mamba-ssm fail during the build

If you encounter any of these erros, try installing using the following commands:

```
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.4.0 # current latest version tag
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ..
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.4 # current latest version tag
CAUSAL_CONV1D_FORCE_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE pip install --no-build-isolation .
```


## Code and Data Availability
All the datasets and models are hosted in Hugging Face 🤗.
* Datasets: https://huggingface.co/datasets/Microsoft/DayhoffDataset
* Models: https://huggingface.co/Microsoft/Dayhoff

## Available models
The available models in Hugging Face 🤗 are:
* 
* 
* 
* 

## Unconditional generation

## Homologue-conditioned generation
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Disclaimer
The [software/model] described in this repository is provided for research and development use only. The [software/model] is not intended for use in clinical decision-making or for any other clinical use, and the performance of model for clinical use has not been established. You bear sole responsibility for any use of this [software/model], including incorporation into any product intended for clinical use. 

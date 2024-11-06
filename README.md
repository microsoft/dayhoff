# Dayhoff

Dayhoff is an Atlas of both protein sequence data and generative language models — a centralized resource that brings together 3.34 billion protein sequences across 1.7 billion clusters of metagenomic and natural protein sequences (GigaRef), 46 million structure-based synthetic sequences (BackboneRef), and 16 million multiple sequence alignments (OpenProteinSet). These models can natively predict zero-shot mutation effects on fitness, scaffold structural motifs by conditioning on evolutionary or structural context, and perform guided generation of novel proteins within specified families. Learning from metagenomic and structure-based synthetic data from the Dayhoff Atlas increased the cellular expression rates of generated proteins, highlighting the real-world value of expanding the scale, diversity, and novelty of protein sequence data. 

The Dayhoff model architecture combines state-space Mamba layers with Transformer self-attention, interleaved with Mixture-of-Experts modules to maximize capacity while preserving efficiency. It natively handles long contexts, allowing both single sequences and unrolled MSAs to be modeled. Trained with an autoregressive objective in both N→C and C→N directions, Dayhoff supports order-agnostic infilling and scales to billions of parameters.

If you use the code, data, models, or results. please cite our [preprint](https://aka.ms/dayhoff/preprint).

<p align="center">
<img src="img/fig1_schematic.png" />
</p>

## Table of Contents
* [Dayhoff](#Dayhoff)
* [Installation](#Installation)
* [Data and Model availability](#Data-and-model-availability)
    * [Datasets](#Datasets)
        * [Training Datasets](#training-datasets)
        * [DayhoffRef](#dayhoffref)
        * [Loading Datasets in HuggingFace](#loading-datasets-in-huggingface)
    * [Models](#models)
        * [170M parameter models](#170m-parameter-models)
        * [3B parameter models](#3b-parameter-models)
* [Unconditional generation](#Unconditional-generation)
* [Homolog-conditioned generation](#Homolog-conditioned-generation)
* [Analysis](#Analysis-scripts)
* [Out-of-scope use cases](#out-of-scope-use-cases)
* [Responsible AI](#responsible-ai-considerations)
* [Contributing](#Contributing)
* [Trademarks](#Trademarks)


## Installation
**Requirements**: 
* PyTorch: 2.2 and above (2.7 recommended)
* CUDA 12.0 and above
* Optionally install Flash Attention 2 following installation instructions here: https://github.com/Dao-AILab/flash-attention

We recommend creating a clean conda environment with Python 3.10

```bash
conda create --name dayhoff python=3.10
```

In that new environment, install PyTorch, mamba-ssm, and causal-conv1d, then install Dayhoff. Optionally, install Flash Attention 2.

```bash
pip install dayhoff

# For bleeding edge: 
pip install git+https://github.com/microsoft/dayhoff.git
```

**Mamba-ssm and causal-conv1d recommendations** 

It is sometimes challenging to properly install these packages just using pip. The following two errors are common when simply using pip install:
* packages installed correctly, but when loading models you get "ValueError: Fast Mamba kernels are not available. Make sure they are installed and that the mamba module is on a CUDA device."
* Package installation of causal-conv1d or mamba-ssm fails during the build

If you encounter any of these errors, try installing using the following commands:

```bash
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

**Docker**

For a fully functional containerized environment without needing to install dependencies manually, you can use the provided Docker image instead:

```bash
docker pull samirchar/dayhoff:latest 
docker run -it samirchar/dayhoff:latest
```

## Data and model availability

All Dayhoff models are available on [Azure AI Foundry](https://aka.ms/dayhoff/foundry)

Additionally, all Dayhoff models are also hosted on [Hugging Face](https://huggingface.co/collections/microsoft/dayhoff-atlas-6866d679465a2685b06ee969) 🤗. All datasets used in the paper, with the exception of OpenProteinSet are available on Hugging Face in three formats: FASTA, Arrow, and JSONL. The PDB files for structures used to generate BackboneRef are available in Parquet format.

GigaRef, BackboneRef, and DayhoffRef are available under [CC BY License](https://creativecommons.org/licenses/by/4.0/)

## Datasets 
### Training datasets
The Dayhoff models were trained on the Dayhoff Atlas, with varying data mixes which include:  

* **[UniRef50](https://www.uniprot.org/)** (**UR50**) - dataset from UniProt, clustered at 50% sequence identity, contains only cluster representatives.
    * _Splits: train (25 GB), test (26 MB), valid (26 MB)_

* **[UniRef90](https://www.uniprot.org/)** (**UR90**) - dataset from UniProt, clustered at 90% sequence identity, contains cluster representatives and members.
    * _Splits: train (83 GB), test (90 MB), valid (87 MB)_


* **GigaRef** (**GR**)– 3.43B protein sequences across 1.7B clusters of metagenomic and natural protein sequences. There are two subsets of gigaref:
    * **GigaRef-clusters** (**GR**) - Only includes cluster representatives and members, no singletons
        * _Splits: train (433 GB), test (22 MB)_
    * **GigaRef-singletons** (**GR-s**) - Only includes singletons
        * _Splits: train (282 GB)_

* **BackboneRef** (**BR**) – 46M structure-derived synthetic sequences from c.a. 240,000 de novo backbones, with three subsets containing 10M sequences each:  
    * **BackboneRef unfiltered** (**BRu**) – 10M sequences randomly sampled from all 46M designs.  
        * _Splits: train (3 GB)_
    * **BackboneRef quality** (**BRq**) – 10M sequences sampled from 127,633 backbones whose average self-consistency RMSD ≤ 2 Å.  
        * _Splits: train(3 GB)_
    * **BackboneRef novelty** (**BRn**) – 10M sequences from 138,044 backbones with a max TM-score < 0.5 to any natural structure.  
        * _Splits: train (3GB)_

* **[OpenProteinSet](https://arxiv.org/abs/2308.05326)** (**HM**) – 16 million precomputed MSAs from 16M sequences in UniClust30 and 140,000 PDB chains. 

### DayhoffRef
Given the potential for generative models to expand the space of proteins and their functions, we used the Dayhoff models to generate DayhoffRef, a PLM-generated database of synthetic protein sequences

* **DayhoffRef**: dataset of 16 million synthetic protein sequences generated by the Dayhoff models: Dayhoff-3b-UR90, Dayhoff-3b-GR-HM, Dayhoff-3b-GR-HM-c, and Dayhoff-170m-UR50-BRn. 
    * _Splits: train (5 GB)_

### Loading datasets in HuggingFace 

Below are some examples on how to load the datasets using `load_dataset` in HuggingFace:

```python
gigaref_clustered_train = load_dataset("microsoft/DayhoffDataset",
                  name="gigaref_no_singletons",
                  split="train")

uniref50_train = load_dataset("microsoft/DayhoffDataset",
                  name="uniref50",
                  split = "train")

backboneref_novelty = load_dataset("microsoft/DayhoffDataset",
                  name="backboneref",
                  split = "BBR_n")
                
dayhoffref = load_dataset("microsoft/DayhoffDataset",
                  name="dayhoffref",
                  split = "train")

```

For the largest datasets, consider using `streaming=True`.

## Models

Weights are available for the following models, as described in the [paper](https://aka.ms/dayhoff/preprint)

### 170M parameter models
* **Dayhoff-170m-UR50**: A 170M parameter model trained on UniRef50 cluster representatives
* **Dayhoff-170m-UR90**: A 170M parameter model trained on UniRef90 members sampled by UniRef50 cluster
* **Dayhoff-170m-GR** : A 170M parameter model trained on members sampled from GigaRef clusters
* **Dayhoff-170m-BRu**: A 170M parameter model trained on UniRef50 cluster representatives and samples from unfiltered BackboneRef
* **Dayhoff-170m-BRq**: A 170M parameter model trained on UniRef50 cluster representatives and samples from quality-filtered BackboneRef
* **Dayhoff-170m-BRn**: A 170M parameter model trained on UniRef50 cluster representatives and samples from novelty-filtered BackboneRef

### 3B parameter models
* **Dayhoff-3b-UR90**: A 3B parameter model trained on UniRef90 members sampled by UniRef50 cluster
* **Dayhoff-3b-GR-HM**: A 3B parameter model trained on members sampled from GigaRef clusters and homologs from OpenProteinSet
* **Dayhoff-3b-GR-HM-c**: A 3B parameter model trained on members sampled from GigaRef clusters and homologs from OpenProteinSet and subsequently cooled using UniRef90 members sampled by UniRef50 cluster and homologs from OpenProteinSet.


## Unconditional generation

For most cases, use [src/generate.py](https://github.com/microsoft/dayhoff/blob/main/src/generate.py) to generate new protein sequences. Below is a sample code to generate 10 sequence with at most 100 residues:

```bash
python src/generate.py --out-dir generations --model 170m-UR50-BBR-n --max-length 100 --n-generations 10 --temp 1.0 --min-p 0.0 --random-seed 1 
```

## Homolog-conditioned generation

The [generate_from_homologs](https://github.com/microsoft/dayhoff/blob/main/src/generate_from_homologs.py) script performs sequence generation conditioned on evolutionarily-related homologous sequences modeled as multiple sequence alignments (MSAs)

The following code specifies the folder where MSAs in fasta format are stored and selects two specific MSAs for conditional generation. The list of MSAs within the MSAs dir can also be specified via an --include-pattern argument.

```bash
python src/generate_from_homologs.py --model 3b-GGR-MSA --msas-dir MSAs --task sequence --out-dir generations --msa-file-names 100220484.fasta 10123434.fasta --temp 1.0 --min-p 0.0 --max-length 768 --random-seed 1 
```

## Analysis scripts

The following scipts were used to conduct analyses described in the paper.

Generation: 
* [generate.py](https://github.com/microsoft/dayhoff/blob/main/analysis/generate.py)

Dataset analysis:
* [clusters.py](https://github.com/microsoft/dayhoff/blob/main/analysis/clusters.py) 
* [gigaref.py](https://github.com/microsoft/dayhoff/blob/main/analysis/gigaref.py)
* [gigaref_clusters.py](https://github.com/microsoft/dayhoff/blob/main/analysis/gigaref_clusters.py) 
* [gigaref_singles.py](https://github.com/microsoft/dayhoff/blob/main/analysis/gigaref_singles.py)
* [gigaref_to_jsonl.py](https://github.com/microsoft/dayhoff/blob/main/analysis/gigaref_to_jsonl.py)
* [create_fasta_sample.py](https://github.com/microsoft/dayhoff/blob/main/analysis/create-fasta-sample.py)
* [extract_test_fastas.py](https://github.com/microsoft/dayhoff/blob/main/analysis/extract_test_fastas.py)
* [plot_metrics.py](https://github.com/microsoft/dayhoff/blob/main/analysis/plot_metrics.py)
* [sample-clustered-splits.py](https://github.com/microsoft/dayhoff/blob/main/analysis/sample-clustered-splits.py)
* [sample_uniref.py](https://github.com/microsoft/dayhoff/blob/main/analysis/sample_uniref.py)


Perplexity:
* [perplexity.py](https://github.com/microsoft/dayhoff/blob/main/analysis/perplexity.py)
* [plot_valid.py](https://github.com/microsoft/dayhoff/blob/main/analysis/plot_valid.py)

Sequence fidelity (via folding and inverse folding):
* [compile_msa_fidelity.py](https://github.com/microsoft/dayhoff/blob/main/analysis/compile_msa_fidelity.py)
* [fidelity.py](https://github.com/microsoft/dayhoff/blob/main/analysis/fidelity.py)
* [plot_fidelity.py](https://github.com/microsoft/dayhoff/blob/main/analysis/plot_fidelity.py)

Distributional embedding analysis (via FPD and PNMMD):
* [embed.py](https://github.com/microsoft/dayhoff/blob/main/analysis/embed.py)
* [fpd.py](https://github.com/microsoft/dayhoff/blob/main/analysis/fpd.py)
* [mmd.py](https://github.com/microsoft/dayhoff/blob/main/analysis/mmd.py)
* [plot_mmd.py](https://github.com/microsoft/dayhoff/blob/main/analysis/plot_mmd.py)

Pfam annotation:
* [pfam.py](https://github.com/microsoft/dayhoff/blob/main/analysis/pfam.py) 

DayhoffRef compilation: 
 * [compile_dayhoffref.py](https://github.com/microsoft/dayhoff/blob/main/analysis/compile_dayhoffref.py)

ProteinGym evals: 
* [xlstm.py](https://github.com/microsoft/dayhoff/blob/main/analysis/xlstm.py)
* [zeroshot.py](https://github.com/microsoft/dayhoff/blob/main/analysis/zeroshot.py)
* [plot_zs.py](https://github.com/microsoft/dayhoff/blob/main/analysis/plot_zs.py)

Scaffolding (Details in README.md in `scaffolding/`): 
* [scaffolding](https://github.com/microsoft/dayhoff/blob/main/analysis/scaffolding)

Evolution guided generation: 
* [pick_test_msas.py](https://github.com/microsoft/dayhoff/blob/main/analysis/pick_test_msas.py)
* [query_from_homologs.py](https://github.com/microsoft/dayhoff/blob/main/analysis/query_from_homologs.py) 
* [evodiff_msa.py](https://github.com/microsoft/dayhoff/blob/main/analysis/evodiff_msa.py)
* [reorg_msas.py](https://github.com/microsoft/dayhoff/blob/main/analysis/reorg_msas.py)

Cas9 evals: 
* [generate_cas9.py](https://github.com/microsoft/dayhoff/blob/main/analysis/generate_cas9.py)
* [compile_cas9_fidelity.py](https://github.com/microsoft/dayhoff/blob/main/analysis/compile_cas9_fidelity.py)

## Out-of-Scope Use Cases

This model should not be used to generate anything that is not a protein sequence or a set of homologuous protein sequences. It is not meant for natural language or other biological sequences, such as DNA sequences.

## Responsible AI Considerations

The intended use of this model is to generate high-quality, realistic, protein sequences or sets of homologous protein sequences. Generations can be designed from scratch or conditioned on partial sequences in both N→C and C→N directions.

Risks and limitations: Not all sequences are guaranteed to be realistic. It remains difficult to generate high-quality sequences with no sequence homology to any natural sequence.

The code and datasets released in this repository are provided for research and development use only. They are not intended for use in clinical decision-making or for any other clinical use, and the performance of these models for clinical use has not been established. You bear sole responsibility for any use of these models, data and software, including incorporation into any product intended for clinical use.

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

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

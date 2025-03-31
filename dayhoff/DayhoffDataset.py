import datasets
import os.path as osp
from torch.utils.data import Dataset
from dayhoff.datasets import UniRefDataset
from dataclasses import dataclass
from typing import Literal

_DESCRIPTION = """\
Dayhoff dataset
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/microsoft/dayhoff/tree/main"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

#TODO: Add citation
_CITATION = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    
}

@dataclass
class SequencesConfig(datasets.BuilderConfig):
        '''Congif for sequence generation'''
        name: str = "sequence"
        max_seq_len: int = 2048
        dataset: Literal["uniref50_202401", "uniref90_202401", "gigaref"] = "uniref50_202401"

@dataclass
class MSAConfig(datasets.BuilderConfig):
        '''Congif for MSA generation'''
        
        name: str = "msa"
        dataset: Literal["uniref50_202401", "gigaref_with_singletons", "gigaref_no_singletons"] = "uniref50_202401" #TODO: complete all possible datasets
        #TODO: Complete all arguments by looking at dataset class, possibly OpenProteinDataset


# Name of the dataset usually matches the script name with CamelCase instead of snake_case
class DayhoffDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""
    
    VERSION = datasets.Version("1.1.0")
    DEFAULT_CONFIG_NAME = "sequence"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        SequencesConfig(version=VERSION, description="sequence generation"),
        MSAConfig(version=VERSION, description="MSA generation"),

    ]

    

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        # Maybe add sequence and MSA configs?
        if self.config.name == "sequence":
            features = datasets.Features(
                {
                    "sequence": datasets.Value("string")
                }
            )

            homepage= "" # TODO: add HF homepage

        if self.config.name == "msa":
             raise NotImplementedError("MSA config not implemented yet") #TODO: Implement MSA config

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            # description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
            
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        if self.config.name == "sequence":
            
            file_paths = dl_manager.download({
                 'consensus':'https://huggingface.co/datasets/samirchar/dayhoff/uniref50_202401/consensus_sample.fasta',
                 'splits':'https://huggingface.co/datasets/samirchar/dayhoff/uniref50_202401/splits.json',
                 'splits':'https://huggingface.co/datasets/samirchar/dayhoff/uniref50_202401/splits_and_offsets.npz'
            })

            print("Data downloaded to: ", file_paths, '\n')
            
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data_dir": "", #TODO: Add data dir
                        "split": "train",
                        "max_len":self.config.max_seq_len, #TODO: can max_len and split be passed as arguments?
                        "split_file":None
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data_dir": "", #TODO: Add data dir
                        "split": "valid",
                        "max_len":self.config.max_seq_len, #TODO: can max_len and split be passed as arguments?
                        "split_file":None
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data_dir": "", #TODO: Add data dir
                        "split": "test",
                        "max_len":self.config.max_seq_len, #TODO: can max_len and split be passed as arguments?
                        "split_file":None
                    },
                ),
            ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, data_dir, split, max_len, split_file):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        if self.config.name == "sequence":
            dataset = UniRefDataset(data_dir = data_dir, split = split, max_len = max_len, split_file = split_file)
            for key, row in enumerate(dataset):
                yield key, {
                    "sequence": row
                }
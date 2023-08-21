import os
from dataclasses import dataclass, field
from typing import Optional

from joblib import dump
from masked_dataset import MaskedDataset
from tokenized_dataset import LineByLineTextDataset
from transformers import HfArgumentParser


def fetch_txt_files(paths_type):
    paths_directory = os.path.normpath(
        os.path.join(curr_dir, "..", "_04_path_splitting", "file_paths")
    )

    file_mapping = {
        "all": "all_paths.txt",
        "train": "train_paths.txt",
        "test": "test_paths.txt",
        "train_test": ["train_paths.txt", "test_paths.txt"],
    }

    # Using a dictionary to store the path_type as the key and its corresponding paths as the value
    txt_files_dict = {}

    file_name = file_mapping.get(paths_type)

    if isinstance(file_name, list):
        for fn in file_name:
            key = fn.split("_")[0]  # Using the filename's prefix as the key
            print(f"Loading {key} paths")

            with open(os.path.join(paths_directory, fn), "r") as f:
                txt_files_dict[key] = f.read().splitlines()
    else:
        print(f"Loading {paths_type} paths")

        with open(os.path.join(paths_directory, file_name), "r") as f:
            txt_files_dict[paths_type] = f.read().splitlines()

    return txt_files_dict


def save_dataset(dataset, base_path, sub_dir, key):
    filename = os.path.join(
        curr_dir, "saved_data", base_path, f"{sub_dir}_{key}_dataset.pkl"
    )
    dump(dataset, filename)


curr_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ScriptArguments:
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})

    do_tokenize_dataset: bool = field(
        default=False, metadata={"help": "Tokenize the dataset"}
    )
    paths: str = field(
        default="train_test",
        metadata={"help": "Which file paths to use: all, train, test, or train_test."},
    )
    tokenizer_dir_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to where the tokenizer should be exported."},
    )

    def __post_init__(self):
        if self.tokenizer_dir_path is None:
            self.tokenizer_dir_path = os.path.normpath(
                os.path.join(
                    curr_dir,
                    "..",
                    "_02_tokenizer_training",
                    "trained_tokenizer_bundle",
                    f"cy{self.model_type}",
                )
            )

    block_size: str = field(default=512)

    do_create_masked_encodings: bool = field(
        default=False, metadata={"help": "Create train and test sets"}
    )
    mlm_type: str = field(
        default="manual",
        metadata={
            "help": "Type of masking to use for masked language modeling. Pass either 'manual' or 'automatic'"
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.do_tokenize_dataset:
        # key is train or test
        # value is a list of paths
        files_list_dict = fetch_txt_files(script_args.paths)
        for key, files_list in files_list_dict.items():
            print(f"Tokezining {key} files")

            tokenized_dataset = LineByLineTextDataset(
                model_type=script_args.model_type,
                tokenizer_dir_path=script_args.tokenizer_dir_path,
                files_list=files_list,
                block_size=script_args.block_size,
            )

            print(f"Saving the tokenized {key} dataset...")
            save_dataset(tokenized_dataset, "encodings", "tokenized", key)
    else:
        print("Skipping dataset tokenization...")

    if script_args.do_create_masked_encodings:
        print("Loading the tokenized datasets...")
        train_dataset, test_dataset = LineByLineTextDataset().load_encodings()

        print("Creating masked datasets...")
        masked_train_dataset = MaskedDataset(
            train_dataset,
            script_args.model_type,
            script_args.mlm_type,
            script_args.mlm_probability,
        )
        masked_test_dataset = MaskedDataset(
            test_dataset,
            script_args.model_type,
            script_args.mlm_type,
            script_args.mlm_probability,
        )

        print("Saved masked datasets...")
        save_dataset(masked_train_dataset, "masked_encodings", "masked", "train")
        save_dataset(masked_test_dataset, "masked_encodings", "masked", "test")
    else:
        print("Skipping masked dataset creation...")


if __name__ == "__main__":
    main()

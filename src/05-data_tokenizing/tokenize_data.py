import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import torch
from dotenv import find_dotenv, load_dotenv
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from transformers import HfArgumentParser

load_dotenv(find_dotenv())


def fetch_txt_files(paths_type):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    paths_directory = os.path.join(
        current_directory, "..", "04-path_splitting", "file_paths"
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


class LineByLineTextDataset(Dataset):
    def __init__(
        self,
        model_type: str,
        tokenizer_dir_path: str,
        files_list: list,
        block_size: str,
    ):
        """
        Taken by https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/
        but modified
        """

        self.examples = []

        print(f"Loading {model_type} tokenizer")
        if model_type == "bert":
            # Load configurations from config.json
            with open(os.path.join(tokenizer_dir_path, "config.json"), "r") as file:
                config = json.load(file)

            tokenizer = BertWordPieceTokenizer(
                os.path.join(tokenizer_dir_path, "vocab.txt"),
                handle_chinese_chars=config["handle_chinese_chars"],
                lowercase=config["do_lower_case"],
            )
        else:  # roberta
            tokenizer = ByteLevelBPETokenizer(
                os.path.join(tokenizer_dir_path, "vocab.json"),
                os.path.join(tokenizer_dir_path, "merges.txt"),
            )
            tokenizer._tokenizer.post_processor = BertProcessing(
                ("</s>", tokenizer.token_to_id("</s>")),
                ("<s>", tokenizer.token_to_id("<s>")),
            )

        tokenizer.enable_truncation(max_length=block_size)

        for file_path in files_list:
            assert os.path.isfile(file_path)

            print("Reading file: ", file_path)
            with open(file_path, encoding="utf-8") as f:
                lines = [
                    line
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]

            print("Running tokenization")
            self.examples.extend(tokenizer.encode_batch(lines))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].ids, dtype=torch.long)


@dataclass
class ScriptArguments:
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})

    paths: str = field(
        default="train_test",
        metadata={"help": "Which file paths to use: all, train, test, or train_test."},
    )

    tokenizer_dir_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to where the tokenizer should be exported."},
    )
    block_size: str = field(default=512)

    def __post_init__(self):
        if self.tokenizer_dir_path is None:
            self.tokenizer_dir_path = (
                f"../02-tokenizer_training/trained_tokenizer_bundle/cy{self.model_type}"
            )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    files_list_dict = fetch_txt_files(script_args.paths)
    for key, files_list in files_list_dict.items():
        print(f"Tokezining {key} files")

        dataset = LineByLineTextDataset(
            model_type=script_args.model_type,
            tokenizer_dir_path=script_args.tokenizer_dir_path,
            files_list=files_list,
            block_size=script_args.block_size,
        )

        # Saving the instance with the respective key in its filename
        with open(f"saved_tokenized_data/tokenized_{key}_dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    main()

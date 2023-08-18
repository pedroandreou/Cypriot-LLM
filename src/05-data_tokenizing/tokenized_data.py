import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from transformers import AutoTokenizer, HfArgumentParser

load_dotenv(find_dotenv())


class LineByLineTextDataset(Dataset):
    def __init__(
        self,
        model_type: str,
        tokenizer_dir_path: str,
        files_list: list,
        block_size: str,
    ):
        self.examples = []

        print(f"Loading {model_type} tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_path)
        # self.tokenizer.model_max_length = block_size
        self.tokenizer.enable_truncation(max_length=block_size)

        for file_path in files_list:
            assert os.path.isfile(file_path)
            print("Creating features from dataset file at: ", file_path)

            print("Reading file: ", file_path)
            with open(file_path, encoding="utf-8") as f:
                lines = [
                    line
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]

            print("Running tokenization")
            self.examples.extend(self.tokenizer.encode_batch(lines))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].ids, dtype=torch.long)


@dataclass
class ScriptArguments:
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})

    allpaths_file_path: str = field(default="./file_paths/all_paths.txt")
    trainpaths_file_path: str = field(default="./file_paths/train_paths.txt")
    testpaths_file_path: str = field(default="./file_paths/test_paths.txt")

    tokenizer_dir_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to where the tokenizer should be exported."},
    )
    block_size: str = field(default=512)

    def __post_init__(self):
        if self.tokenizer_dir_path is None:
            self.tokenizer_dir_path = f"./trained_tokenizer_bundle/cy{self.model_type}"


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Get list of files to use
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_reformatting_directory = os.path.join(
        current_directory, "..", "02-data_reformatting"
    )
    txt_files = glob.glob(os.path.join(data_reformatting_directory, "*.txt"))

    dataset = LineByLineTextDataset(
        model_type=script_args.model_type,
        tokenizer_dir_path=script_args,
        files_list=script_args,
        block_size=script_args.block_size,
    )


if __name__ == "__main__":
    main()

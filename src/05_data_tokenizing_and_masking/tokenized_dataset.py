import json
import os

import torch
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset


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
        self.model_type = model_type
        self.examples = []

        print(f"Loading {self.model_type} tokenizer")
        if self.model_type == "bert":
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
        tokenizer.enable_padding(length=block_size)

        for file_path in files_list:
            if not os.path.isfile(file_path):
                print(f"Problem with path: {file_path}")

            print("Reading file: ", file_path)
            with open(file_path, encoding="utf-8") as f:
                lines = [
                    line
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]

            print("Running tokenization")
            self.examples.extend(tokenizer.encode_batch(lines))

    def __repr__(self):
        tokenizer_type = (
            "BertWordPieceTokenizer"
            if self.model_type == "bert"
            else "ByteLevelBPETokenizer"
        )
        return f"<LineByLineTextDataset: ModelType={self.model_type}, TokenizerType={tokenizer_type}, NumExamples={len(self.examples)}>"

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].ids, dtype=torch.long)

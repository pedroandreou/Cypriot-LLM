import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src._02_tokenizer_training.main import TokenizerWrapper
from src.utils.common_utils import echo_with_color

"""
A dataloader for accessing the tokenized dataset's token ids where each dataloader's index is returned in a different tensor.
"""

curr_dir = os.path.dirname(os.path.realpath(__file__))


class TokenizedDataset(Dataset):
    def __init__(
        self,
        model_type: str = None,
        tokenizer_version: int = None,
        files_list: list = None,
        block_size: str = None,
    ):
        self.encodings = None

        if all(
            arg is None
            for arg in (model_type, tokenizer_version, files_list, block_size)
        ):
            self.default_constructor()
        else:
            self.parameterized_constructor(
                model_type, tokenizer_version, files_list, block_size
            )

    def default_constructor(self):
        print(
            "Using default constructor. This instance of TokenizedDataset class is meant for loading data only."
        )
        pass

    def parameterized_constructor(
        self, model_type, tokenizer_version, files_list, block_size
    ):
        self.model_type = model_type
        self.tokenizer_version = tokenizer_version
        self.files_list = files_list
        self.block_size = block_size

        self._create_dataset()

    def _create_dataset(self):
        """
        Taken by https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/
        but modified
        """

        echo_with_color(f"Loading {self.model_type} tokenizer", color="bright_yellow")
        tokenizer = TokenizerWrapper().load_tokenizer(
            self.model_type,
            self.tokenizer_version,
            self.block_size,
        )

        examples = []
        tqdm_iterator = tqdm(self.files_list, desc="Tokenizing files")
        for file_path in tqdm_iterator:
            tqdm_iterator.set_description(f"Tokenizing file: {file_path}")

            if not os.path.isfile(file_path):
                print(f"Problem with path: {file_path}")

            with open(file_path, encoding="utf-8") as f:
                lines = [
                    line
                    for line in f.read().splitlines()
                    if (len(line) > 0 and not line.isspace())
                ]
            examples.extend(tokenizer.encode_batch(lines))

        # Prepare the dataset
        self.encodings = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for example in examples:
            current_tensor_labels = example.clone()
            current_tensor_mask = (current_tensor_labels != 0).long()
            current_tensor_input_ids = current_tensor_labels.detach().clone()

            current_tensor_encodings = {
                "input_ids": current_tensor_input_ids,
                "attention_mask": current_tensor_mask,
                "labels": current_tensor_labels,
            }

            for key in current_tensor_encodings:
                self.encodings[key].append(current_tensor_encodings[key])

        for key in self.encodings:
            self.encodings[key] = torch.stack(self.encodings[key])

    def __repr__(self):
        tokenizer_type = (
            "BertWordPieceTokenizer"
            if self.model_type == "bert"
            else "ByteLevelBPETokenizer"
        )
        return f"<TokenizedDataset: ModelType={self.model_type}, TokenizerType={tokenizer_type}, NumExamples={len(self.examples)}>"

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.encodings.items()}

    @staticmethod
    def load_encodings(model_type: str, encodings_version: int):
        def get_dataset_path(set_type: str, encodings_version: int):
            folder_name = os.path.join(
                "encodings", f"cy{model_type}", f"encodings_v{encodings_version}"
            )
            filename = f"tokenized_{set_type}_dataset.pth"

            return os.path.join(curr_dir, folder_name, filename)

        train_dataset_path = get_dataset_path("train", encodings_version)
        train_dataset = torch.load(train_dataset_path)

        test_dataset_path = get_dataset_path("test", encodings_version)
        test_dataset = torch.load(test_dataset_path)

        return train_dataset, test_dataset

import os

import torch
import typer
from torch.utils.data import Dataset
from tqdm import tqdm

from src._02_tokenizer_training.main import TokenizerWrapper


class LineByLineTextDataset(Dataset):
    def __init__(
        self,
        model_type: str = None,
        tokenizer_dir_path: str = None,
        files_list: list = None,
        block_size: str = None,
    ):
        if (
            model_type is None
            and tokenizer_dir_path is None
            and files_list is None
            and block_size is None
        ):
            self.default_constructor()
        else:
            self.parameterized_constructor(
                model_type, tokenizer_dir_path, files_list, block_size
            )

    def default_constructor(self):
        print(
            "Using default constructor. This instance is meant for loading data only."
        )
        self.examples = []
        self.model_type = None

    def parameterized_constructor(
        self, model_type, tokenizer_dir_path, files_list, block_size
    ):
        """
        Taken by https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/
        but modified
        """

        self.model_type = model_type
        self.examples = []

        typer.echo(
            typer.style(
                f"Loading {self.model_type} tokenizer", fg=typer.colors.BRIGHT_YELLOW
            )
        )
        tokenizer = TokenizerWrapper(
            self.model_type,
            tokenizer_dir_path,
            files_list,
            block_size,
        ).load_tokenizer()

        # Wrapping files_list with tqdm to display the progress bar
        tqdm_iterator = tqdm(files_list, desc="Tokenizing files")
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
            self.examples.extend(tokenizer.encode_batch(lines))

    def __repr__(self):
        tokenizer_type = (
            "BertWordPieceTokenizer"
            if self.model_type == "bert"
            else "ByteLevelBPETokenizer"
        )
        return f"<LineByLineTextDataset: ModelType={self.model_type}, TokenizerType={tokenizer_type}, NumExamples={len(self.examples)}>"

    def __len__(self):
        if not self.examples:
            print("Warning: Dataset not initialized. Returning length 0.")
            return 0
        return len(self.examples)

    def __getitem__(self, i):
        if not self.examples:
            print("Warning: Dataset not initialized. Returning empty item.")
            return {}
        return torch.tensor(self.examples[i].ids, dtype=torch.long)

    def load_encodings(self):
        def get_dataset_path(set_type):
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            folder_name = "encodings"
            filename = f"tokenized_{set_type}_dataset.pth"

            return os.path.join(curr_dir, folder_name, filename)

        train_dataset_path = get_dataset_path("train")
        train_dataset = torch.load(train_dataset_path)

        test_dataset_path = get_dataset_path("test")
        test_dataset = torch.load(test_dataset_path)

        return train_dataset, test_dataset

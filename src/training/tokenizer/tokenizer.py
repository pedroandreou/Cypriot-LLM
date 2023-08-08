import json
import os

import typer
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer


class TokenizerWrapper:
    def __init__(
        self,
        train_paths: list,
        tokenizer_path: str,
        model_type: str = "bert",
        vocab_size=30_522,
        max_length: str = 512,
    ):
        self.train_paths = train_paths
        self.tokenizer_path = tokenizer_path
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.max_length = max_length

        typer.echo("Initializing the tokenizer...")
        self.tokenizer = (
            BertWordPieceTokenizer(
                clean_text=True,
                handle_chinese_chars=False,
                strip_accents=False,
                lowercase=False,
            )
            if self.model_type.lower() == "bert"
            else ByteLevelBPETokenizer()
        )

        typer.echo("Training the tokenizer...")
        self.train_tokenizer()

        typer.echo("Saving the tokenizer...")
        self.save_tokenizer()

    def train_tokenizer(self):
        self.special_tokens = (
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
            if self.model_type.lower() == "bert"
            else ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )

        train_params = {
            "files": self.train_paths[:5],  # CHANGE THIS
            "vocab_size": self.vocab_size,  # number of tokens in our tokenizer
            "special_tokens": self.special_tokens,
        }

        if self.model_type.lower() == "bert":
            train_params[
                "limit_alphabet"
            ] = 1000  # maximum number of different characters
            train_params[
                "wordpieces_prefix"
            ] = "##"  # the prefix added to pieces of words

        else:  # roberta
            train_params[
                "min_frequency"
            ] = 2  # minimum frequency for a pair of tokens to be merged

        self.tokenizer.train(**train_params)
        self.tokenizer.enable_truncation(self.max_length)

    def save_tokenizer(self):
        # Save the tokenizer
        self.tokenizer.save_model(self.tokenizer_path)

        #   ## bert
        #   # ['/content/drive/MyDrive/Uni/Masters/Thesis/project/cybert/tokenizer/vocab.txt']

        #   # # ['/content/drive/MyDrive/Uni/Masters/Thesis/project/cybert/tokenizer/config.json']
        if self.model_type == "bert":
            # dumping some of the tokenizer config to config file,
            # including special tokens, whether to lower case and the maximum sequence length
            with open(os.path.join(self.tokenizer_path, "config.json"), "w") as f:
                tokenizer_cfg = {
                    "do_lower_case": True,
                    "unk_token": "[UNK]",
                    "sep_token": "[SEP]",
                    "pad_token": "[PAD]",
                    "cls_token": "[CLS]",
                    "mask_token": "[MASK]",
                    "model_max_length": self.max_length,
                    "max_len": self.max_length,
                }
                json.dump(tokenizer_cfg, f)

        ## roberta
        # # ['/content/drive/MyDrive/Uni/Masters/Thesis/project/cyroberta/tokenizer/vocab.json',
        # # ['/content/drive/MyDrive/Uni/Masters/Thesis/project/cyroberta/tokenizer/merges.txt']

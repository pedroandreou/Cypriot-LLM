import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import BertTokenizerFast, RobertaTokenizer


class TokenizerWrapper:
    def __init__(
        self,
        base_path="/content/drive/MyDrive/Uni/Masters/Thesis",
        model_type="bert",
        vocab_size=30_522,
        max_length=512,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Create train and test sets
        self.all_paths = [
            str(x) for x in Path(f"{base_path}/cleaned_files/").glob("*.txt")
        ]
        self.train_paths, self.test_paths = train_test_split(
            self.all_paths, test_size=0.2
        )

        # Initialize tokenizer
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
        # Train tokenizer
        self.train_tokenizer()

        # Set paths
        dir_path = f"{base_path}/Project/cy{self.model_type}"
        self.tokenizer_path = f"{dir_path}/tokenizer"

        # Create dirs if do not exist
        self.create_directory_if_does_not_exist(dir_path)
        self.create_directory_if_does_not_exist(self.tokenizer_path)

        # Export tokenizer in a dir called either 'cybert' or 'cyroberta'
        self.save_tokenizer()

    def create_directory_if_does_not_exist(self, path):
        # Create the directory if not already there
        if not os.path.exists(path):
            os.makedirs(path)

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

    def decode_input_ids_to_string(self, ids):
        decoded_string = self.tokenizer.decode(ids)
        return decoded_string

    def convert_input_ids_to_tokens(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def read_roberta_tokenizer_files(self):
        with open(f"{self.tokenizer_path}/merges.txt", "r") as file:
            content = file.read()
        print(f"The merges file is:\n{content}\n\n")

        with open(f"{self.tokenizer_path}/vocab.json", "r") as file:
            content = json.load(file)
        print(f"The vocab file is:\n{content}\n\n")

    def get_paths(self):
        """Return all paths"""
        return self.all_paths, self.train_paths, self.test_paths

    def get_tokenizer(self):
        if self.model_type == "bert":
            loaded_tokenizer = BertTokenizerFast.from_pretrained(
                self.tokenizer_path, max_len=512
            )
        else:  # roberta
            loaded_tokenizer = RobertaTokenizer.from_pretrained(
                self.tokenizer_path, max_len=512
            )

        return loaded_tokenizer

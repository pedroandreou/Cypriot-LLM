import json
import os
from glob import glob

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from src.utils.common_utils import echo_with_color, get_new_subdirectory_path


class TokenizerWrapper:
    def __init__(
        self,
        tokenizer_type: str = None,
        block_size: int = None,
        clean_text: bool = None,
        handle_chinese_chars: bool = None,
        strip_accents: bool = None,
        lowercase: bool = None,
        vocab_size: int = None,
        limit_alphabet: int = None,
        min_frequency: int = None,
    ):
        if all(
            arg is None
            for arg in (
                tokenizer_type,
                block_size,
                clean_text,
                handle_chinese_chars,
                strip_accents,
                lowercase,
                vocab_size,
                limit_alphabet,
                min_frequency,
            )
        ):
            self.default_constructor()
        else:
            self.parameterized_constructor(
                tokenizer_type,
                block_size,
                clean_text,
                handle_chinese_chars,
                strip_accents,
                lowercase,
                vocab_size,
                limit_alphabet,
                min_frequency,
            )

    def default_constructor(self):
        print(
            "Using default constructor. This instance of TokenizerWrapper class is meant for loading the tokenizer only."
        )

    def parameterized_constructor(
        self,
        tokenizer_type,
        block_size,
        clean_text,
        handle_chinese_chars,
        strip_accents,
        lowercase,
        vocab_size,
        limit_alphabet,
        min_frequency,
    ):
        self.tokenizer_type = tokenizer_type
        self.block_size = block_size
        self.clean_text = clean_text
        self.handle_chinese_chars = handle_chinese_chars
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.limit_alphabet = limit_alphabet
        self.min_frequency = min_frequency

        filepaths_dir = os.path.join(
            curr_dir,
            "..",
            "_01_data_preprocessing",
            "_04_csv_to_txt_conversion",
            "cleaned_files",
        )
        self.filepaths = list(glob(os.path.join(filepaths_dir, "*.txt")))

        # Create output directory
        tokenizer_dir_path_w_tokenizer_type = os.path.join(
            tokenizer_dir_path, f"cy{self.tokenizer_type}"
        )
        self.tokenizer_dir_path_w_tokenizer_type_n_version = get_new_subdirectory_path(
            tokenizer_dir_path_w_tokenizer_type, "tokenizer"
        )

    def get_tokenizer_and_train_args(self):
        common_tokenizer_args = {
            "lowercase": self.lowercase,
        }
        common_train_args = {
            "files": self.filepaths,
            "vocab_size": self.vocab_size,
        }

        if self.tokenizer_type == "bert":
            return (
                BertWordPieceTokenizer(
                    **common_tokenizer_args,
                    clean_text=self.clean_text,
                    handle_chinese_chars=self.handle_chinese_chars,
                    strip_accents=self.strip_accents,
                ),
                {
                    **common_train_args,
                    "limit_alphabet": self.limit_alphabet,
                    "wordpieces_prefix": "##",
                    "special_tokens": [
                        "[PAD]",
                        "[UNK]",
                        "[CLS]",
                        "[SEP]",
                        "[MASK]",
                        # "<S>",
                        # "<T>",
                    ],
                },
            )
        return (
            ByteLevelBPETokenizer(**common_tokenizer_args),
            {
                **common_train_args,
                "min_frequency": self.min_frequency,
                "special_tokens": ["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
            },
        )

    def train_tokenizer(self):

        echo_with_color(
            f"Initalizing the {self.tokenizer_type}'s tokenizer...", color="black"
        )
        tokenizer, train_args = self.get_tokenizer_and_train_args()

        echo_with_color(
            f"Training the {self.tokenizer_type}'s tokenizer...", color="black"
        )
        tokenizer.train(**train_args)

        echo_with_color(
            f"Saving the {self.tokenizer_type}'s tokenizer files...", color="black"
        )
        tokenizer.save_model(self.tokenizer_dir_path_w_tokenizer_type_n_version)

        # Save WordPiece's config json file to disk
        # as it is not saved automatically
        if self.tokenizer_type == "WP":
            config_path = os.path.join(
                self.tokenizer_dir_path_w_tokenizer_type_n_version, "config.json"
            )
            with open(config_path, "w") as f:
                tokenizer_cfg = {
                    # "model_type": "bert", # For AutoTokenizer.from_pretrained
                    "handle_chinese_chars": self.handle_chinese_chars,
                    "do_lower_case": self.lowercase,
                    "strip_accents": self.strip_accents,
                    "unk_token": "[UNK]",
                    "sep_token": "[SEP]",
                    "pad_token": "[PAD]",
                    "cls_token": "[CLS]",
                    "mask_token": "[MASK]",
                    "model_max_length": self.block_size,
                    "max_len": self.block_size,
                }
                json.dump(tokenizer_cfg, f)

    @staticmethod
    def get_tokenizer_paths(tokenizer_type: str, tokenizer_version: int):
        """
        This method is used to get the paths to the tokenizer's files.
        Essentially it should a nested function of the 'load_tokenizer' function below but since our tokenizer is trained using the 'tokenizers' library
        but the Inference Pipeline is using the 'transformers' library, we separated the function to be able to use it in both cases.
        """
        paths = []

        try:
            specific_tokenizer_dir_path = os.path.join(
                tokenizer_dir_path,
                f"cy{tokenizer_type}",
                f"tokenizer_v{tokenizer_version}",
            )
        except FileNotFoundError:
            print(
                f"Directory '{specific_tokenizer_dir_path}' does not exist. Please ensure the provided tokenizer version is correct."
            )

        if tokenizer_type == "WP":
            config_path = os.path.join(specific_tokenizer_dir_path, "config.json")
            vocab_path = os.path.join(specific_tokenizer_dir_path, "vocab.txt")
            paths.append(config_path)
            paths.append(vocab_path)

        else:  # roberta
            vocab_path = os.path.join(specific_tokenizer_dir_path, "vocab.json")
            merges_path = os.path.join(specific_tokenizer_dir_path, "merges.txt")
            paths.append(vocab_path)
            paths.append(merges_path)

        return paths

    @staticmethod
    def load_tokenizer(tokenizer_type: str, tokenizer_version: int, block_size: int):
        """
        Taken by https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/
        but modified
        """

        if tokenizer_type == "WP":
            config_path, vocab_path = TokenizerWrapper.get_tokenizer_paths(
                tokenizer_type, tokenizer_version
            )

            # Load configurations
            with open(config_path, "r") as file:
                config = json.load(file)

            # Load tokenizer
            tokenizer = BertWordPieceTokenizer(
                vocab_path,
                handle_chinese_chars=config["handle_chinese_chars"],
                lowercase=config["do_lower_case"],
            )

        else:  # roberta
            vocab_path, merges_path = TokenizerWrapper.get_tokenizer_paths(
                tokenizer_type, tokenizer_version
            )

            # Load tokenizer
            tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
            tokenizer._tokenizer.post_processor = BertProcessing(
                ("</s>", tokenizer.token_to_id("</s>")),
                ("<s>", tokenizer.token_to_id("<s>")),
            )

        tokenizer.enable_truncation(max_length=block_size)
        tokenizer.enable_padding(length=block_size)

        return tokenizer


curr_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_dir_path = os.path.join(curr_dir, "trained_tokenizer_bundle")


def main(
    tokenizer_type: str,
    block_size: int,
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
    vocab_size: int,
    limit_alphabet: int,
    min_frequency: int = None,
    do_push_tokenizer_to_hub=None,
):

    TokenizerWrapper(
        tokenizer_type=tokenizer_type,
        block_size=block_size,
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        strip_accents=strip_accents,
        lowercase=lowercase,
        vocab_size=vocab_size,
        limit_alphabet=limit_alphabet,
        min_frequency=min_frequency,
    ).train_tokenizer()

    if do_push_tokenizer_to_hub:
        from utils.hub_pusher import push_tokenizer

        # Normally we push the whole tokenizer
        # but since we are using 'tokenizers' instead of 'transformers' library
        # we need to push the files manually
        tokenizer_paths = TokenizerWrapper().get_tokenizer_paths(tokenizer_type)

        push_tokenizer(
            curr_dir,
            tokenizer_paths,
        )

    else:
        echo_with_color(
            "Skipping push to the hub.",
            color="black",
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script parameters.")

    # Basic arguments
    parser.add_argument(
        "--tokenizer_type", type=str, default="WP", help="Tokenizer type to use."
    )
    parser.add_argument("--block_size", type=int, default=512, help="Block size.")

    # Tokenizer parameters
    parser.add_argument(
        "--clean_text",
        type=bool,
        default=True,
        help="Clean text during tokenization. Accepts: True/False. Default is True.",
    )
    parser.add_argument(
        "--handle_chinese_chars",
        type=bool,
        default=True,
        help="Handle Chinese characters during tokenization. Accepts: True/False. Default is True.",
    )
    parser.add_argument(
        "--strip_accents",
        type=bool,
        default=True,
        help="Strip accents during tokenization. Accepts: True/False. Default is True.",
    )
    parser.add_argument(
        "--lowercase",
        type=bool,
        default=True,
        help="Convert to lowercase during tokenization. Accepts: True/False. Default is True.",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Vocabulary size."
    )
    parser.add_argument(
        "--limit_alphabet",
        type=int,
        default=1000,
        help="Maximum different characters to keep.",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for a token to be included in the vocabulary.",
    )

    # Flags for actions
    parser.add_argument(
        "--do_push_tokenizer_to_hub",
        type=bool,
        default=False,
        help="Whether or not to push tokenizer to the hub. Provide a boolean value (e.g., True/False, Yes/No).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_arguments()

    main(
        args.tokenizer_type,
        args.block_size,
        args.clean_text,
        args.handle_chinese_chars,
        args.strip_accents,
        args.lowercase,
        args.vocab_size,
        args.limit_alphabet,
        args.min_frequency,
        args.do_push_tokenizer_to_hub,
    )

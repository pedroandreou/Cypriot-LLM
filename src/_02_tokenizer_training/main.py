import json
import os
from glob import glob

import typer
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


class TokenizerWrapper:
    def __init__(
        self,
        model_type: str,
        tokenizer_dir_path: str,
        filepaths_dir: str,
        block_size: int,
    ):
        self.model_type = model_type
        self.tokenizer_dir_path = tokenizer_dir_path
        self.filepaths_dir = filepaths_dir
        self.block_size = block_size

    def train_tokenizer(self):
        typer.echo(
            typer.style(
                "Loading configurations from the JSON file...", fg=typer.colors.BLACK
            )
        )
        config_path = os.path.join(
            curr_dir, "initial_configs", f"{self.model_type}_config.json"
        )
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        typer.echo(
            typer.style(
                f"Initializing the {self.model_type}'s tokenizer...",
                fg=typer.colors.BLACK,
            )
        )
        if self.model_type == "bert":
            tokenizer = BertWordPieceTokenizer(
                clean_text=config_dict["clean_text"],
                handle_chinese_chars=config_dict["handle_chinese_chars"],
                strip_accents=config_dict["strip_accents"],
                lowercase=config_dict["lowercase"],
            )
        else:
            tokenizer = ByteLevelBPETokenizer()

        filepaths = list(glob(os.path.join(self.filepaths_dir, "*.txt")))
        typer.echo(
            typer.style(
                f"Training the {self.model_type}'s tokenizer...", fg=typer.colors.BLACK
            )
        )
        if self.model_type == "bert":
            tokenizer.train(
                files=filepaths,
                vocab_size=config_dict["vocab_size"],
                limit_alphabet=config_dict["limit_alphabet"],
                wordpieces_prefix=config_dict["wordpieces_prefix"],
                special_tokens=config_dict["special_tokens"],
            )
        else:
            tokenizer.train(
                files=filepaths,
                min_frequency=config_dict["min_frequency"],
            )

        typer.echo(
            typer.style(
                f"Saving the {self.model_type}'s tokenizer...", fg=typer.colors.BLACK
            )
        )
        tokenizer.save_model(self.tokenizer_dir_path)

        if self.model_type == "bert":

            config_path = os.path.join(self.tokenizer_dir_path, "config.json")
            with open(config_path, "w") as f:
                tokenizer_cfg = {
                    # "model_type": "bert", # For AutoTokenizer.from_pretrained
                    "handle_chinese_chars": False,
                    "do_lower_case": True,
                    "unk_token": "[UNK]",
                    "sep_token": "[SEP]",
                    "pad_token": "[PAD]",
                    "cls_token": "[CLS]",
                    "mask_token": "[MASK]",
                    "model_max_length": self.block_size,
                    "max_len": self.model_type,
                }
                json.dump(tokenizer_cfg, f)

    def get_tokenizer_paths(self):
        paths = []
        if self.model_type == "bert":
            config_path = os.path.join(self.tokenizer_dir_path, "config.json")
            vocab_path = os.path.join(self.tokenizer_dir_path, "vocab.txt")
            paths.append(config_path)
            paths.append(vocab_path)

        else:  # roberta
            vocab_path = os.path.join(self.tokenizer_dir_path, "vocab.json")
            merges_path = os.path.join(self.tokenizer_dir_path, "merges.txt")
            paths.append(vocab_path)
            paths.append(merges_path)

        return paths

    def load_tokenizer(self):
        """
        Taken by https://zablo.net/blog/post/training-roberta-from-scratch-the-missing-guide-polish-language-model/
        but modified
        """
        if self.model_type == "bert":
            config_path, vocab_path = self.get_tokenizer_paths()

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
            vocab_path, merges_path = self.get_tokenizer_paths()

            # Load tokenizer
            tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
            tokenizer._tokenizer.post_processor = BertProcessing(
                ("</s>", tokenizer.token_to_id("</s>")),
                ("<s>", tokenizer.token_to_id("<s>")),
            )

        tokenizer.enable_truncation(max_length=self.block_size)
        tokenizer.enable_padding(length=self.block_size)

        return tokenizer


curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(
    model_type,
    cleaned_files_dir_path,
    block_size,
    do_push_tokenizer_to_hub,
    do_login_first_time,
    huggingface_token,
    huggingface_repo_name,
):

    tokenizer_dir_path = os.path.join(
        curr_dir, "trained_tokenizer_bundle", f"cy{model_type}"
    )

    TokenizerWrapper(
        model_type=model_type,
        tokenizer_dir_path=tokenizer_dir_path,
        filepaths_dir=cleaned_files_dir_path,
        block_size=block_size,
    ).train_tokenizer()

    if do_push_tokenizer_to_hub:
        from src.hub_pusher import push_tokenizer

        tokenizer_paths = TokenizerWrapper(
            model_type=model_type,
            tokenizer_dir_path=tokenizer_dir_path,
            filepaths_dir=cleaned_files_dir_path,
            block_size=block_size,
        ).get_tokenizer_paths()

        push_tokenizer(
            curr_dir,
            tokenizer_paths,
            do_login_first_time,
            huggingface_token,
            huggingface_repo_name,
        )

    else:
        typer.echo(typer.style("Skipping push to hub.", fg=typer.colors.BLACK))


if __name__ == "__main__":
    import argparse

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
    )

    default_cleaned_files_path = os.getenv("CLEANED_FILES_DIR_PATH")
    parser.add_argument(
        "--cleaned_files_dir_path", type=str, default=default_cleaned_files_path
    )

    parser.add_argument(
        "--block_size", type=int, default=512, help="Define the block size."
    )

    parser.add_argument(
        "--do_push_tokenizer_to_hub",
        action="store_true",
        help="Enable or disable pushing tokenizer to hub.",
    )

    parser.add_argument(
        "--do_login_first_time",
        action="store_true",
        help="Toggle first-time login. Credentials will be cached after the initial login to the hub.",
    )

    default_huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=default_huggingface_token,
        help="Token for HuggingFace.",
    )

    default_huggingface_repo_name = os.getenv("HUGGINGFACE_REPO_NAME")
    parser.add_argument(
        "--huggingface_repo_name",
        type=str,
        default=default_huggingface_repo_name,
        help="Name of the HuggingFace repo.",
    )

    args = parser.parse_args()

    main(
        args.model_type,
        args.cleaned_files_dir_path,
        args.block_size,
        args.do_push_tokenizer_to_hub,
        args.do_login_first_time,
        args.huggingface_token,
        args.huggingface_repo_name,
    )

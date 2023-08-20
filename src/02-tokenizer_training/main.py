import json
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import HfArgumentParser

load_dotenv(find_dotenv())


class TokenizerWrapper:
    def __init__(
        self,
        model_type: str,
        tokenizer_path: str,
        filepaths_dir: str,
    ):

        print("Loading configurations from the JSON file...")
        with open(f"initial_configs/{model_type}_config.json", "r") as f:
            config_dict = json.load(f)

        print(f"Initializing the {model_type}'s tokenizer...")
        if model_type == "bert":
            tokenizer = BertWordPieceTokenizer(
                clean_text=config_dict["clean_text"],
                handle_chinese_chars=config_dict["handle_chinese_chars"],
                strip_accents=config_dict["strip_accents"],
                lowercase=config_dict["lowercase"],
            )
        else:
            tokenizer = ByteLevelBPETokenizer()

        filepaths = list(glob(os.path.join(filepaths_dir, "*.txt")))
        print(f"Training the {model_type}'s tokenizer...")
        if model_type == "bert":
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

        print(f"Saving the {model_type}'s tokenizer...")
        tokenizer.save_model(tokenizer_path)

        if model_type == "bert":
            # dumping some of the tokenizer config to config file,
            # including special tokens, whether to lower case and the maximum sequence length
            with open(os.path.join(tokenizer_path, "config.json"), "w") as f:
                tokenizer_cfg = {
                    # "model_type": "bert", # For AutoTokenizer.from_pretrained
                    "handle_chinese_chars": False,
                    "do_lower_case": True,
                    "unk_token": "[UNK]",
                    "sep_token": "[SEP]",
                    "pad_token": "[PAD]",
                    "cls_token": "[CLS]",
                    "mask_token": "[MASK]",
                    "model_max_length": 512,
                    "max_len": 512,
                }
                json.dump(tokenizer_cfg, f)


@dataclass
class ScriptArguments:
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})

    cleaned_files_dir_path: str = field(default=os.getenv("CLEANED_FILES_DIR_PATH"))

    do_train_tokenizer: bool = field(default=False)
    tokenizer_dir_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to where the tokenizer should be exported."},
    )
    do_push_tokenizer_to_hub: bool = field(
        default=False, metadata={"help": "Enable or disable pushing tokenizer to hub."}
    )

    def __post_init__(self):
        if self.tokenizer_dir_path is None:
            self.tokenizer_dir_path = f"./trained_tokenizer_bundle/cy{self.model_type}"


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.do_train_tokenizer:
        print("Training a tokenizer from scratch...")

        TokenizerWrapper(
            model_type=script_args.model_type,
            tokenizer_path=script_args.tokenizer_dir_path,
            filepaths_dir=script_args.cleaned_files_dir_path,
        )

    else:
        print("Skipping the training of a tokenizer...")


if __name__ == "__main__":
    main()

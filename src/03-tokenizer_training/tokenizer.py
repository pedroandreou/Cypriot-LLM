import json
import os

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer


class TokenizerWrapper:
    def __init__(
        self,
        filepaths: list,
        tokenizer_path: str,
        model_type: str,
    ):
        self.filepaths = filepaths
        self.tokenizer_path = tokenizer_path
        self.model_type = model_type

        print("Loading configurations from the JSON file...")
        with open(f"initial_configs/{self.model_type}_config.json", "r") as f:
            config_dict = json.load(f)

        print("Initializing the tokenizer...")
        if self.model_type == "bert":
            self.tokenizer = BertWordPieceTokenizer(
                clean_text=config_dict["clean_text"],
                handle_chinese_chars=config_dict["handle_chinese_chars"],
                strip_accents=config_dict["strip_accents"],
                lowercase=config_dict["lowercase"],
            )
        else:
            self.tokenizer = ByteLevelBPETokenizer()

        print("Training the tokenizer...")
        if model_type == "bert":
            self.tokenizer.train(
                files=self.filepaths,
                vocab_size=config_dict["vocab_size"],
                limit_alphabet=config_dict["limit_alphabet"],
                wordpieces_prefix=config_dict["wordpieces_prefix"],
                special_tokens=config_dict["special_tokens"],
            )
        else:
            self.tokenizer.train(
                files=self.filepaths,
                min_frequency=config_dict["min_frequency"],
            )
        # self.tokenizer.enable_truncation(self.max_length)

        print("Saving the tokenizer...")
        self.tokenizer.save_model(self.tokenizer_path)

        if self.model_type == "bert":
            # dumping some of the tokenizer config to config file,
            # including special tokens, whether to lower case and the maximum sequence length
            with open(os.path.join(self.tokenizer_path, "config.json"), "w") as f:
                tokenizer_cfg = {
                    "model_type": "bert",
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

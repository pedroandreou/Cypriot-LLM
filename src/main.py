import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from transformers import HfArgumentParser

from src._01_data_preprocessing.data_cleaner.main import main as clean_data
from src._01_data_preprocessing.doc_merge_to_csv.docs_to_csv import main as merge_docs
from src._01_data_preprocessing.export_csv_docs_to_txt_files import (
    main as export_csv_to_txt_files,
)
from src._01_data_preprocessing.file_analysis_helpers.calculate_file_capacities import (
    main as calculate_file_capacities,
)
from src._01_data_preprocessing.file_analysis_helpers.compare_token_counts import (
    main as compare_token_counts,
)
from src._02_tokenizer_training.main import main as train_tokenizer
from src._03_data_reformatting.reformatter import main as reformat_files
from src._04_path_splitting.main import main as split_paths
from src._05_data_tokenizing.main import main as tokenize_files
from src._06_data_masking.main import main as create_masked_encodings
from src._07_model_training.main import main as train_model
from src._08_inferencing.pipeline import main as infer
from src.utils.common_utils import echo_with_color

load_dotenv(find_dotenv())


curr_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ScriptArguments:
    ### MERGING DATA ###
    do_merge_docs: bool = field(
        default=False,
        metadata={"help": "Enable or disable data merging into a single CSV file."},
    )
    data_path: Optional[str] = field(
        default=os.getenv("DATASET_DIR_PATH"),
        metadata={"help": "Path to the dataset directory."},
    )

    ### CLEANING DATA ###
    do_clean_data: bool = field(
        default=False, metadata={"help": "Enable or disable data cleaning."}
    )
    do_push_dataset_to_hub: bool = field(
        default=False, metadata={"help": "Enable or disable push to hub."}
    )

    ### DO FILE ANALYSIS ###
    do_file_analysis: bool = field(
        default=False, metadata={"help": "Enable or disable file analysis."}
    )

    ### EXPORTING CSV TO TXT FILES ###
    do_export_csv_to_txt_files: bool = field(
        default=False,
        metadata={"help": "Enable or disable export of CSV to txt files."},
    )
    do_load_dataset_from_hub: bool = field(
        default=False,
        metadata={"help": "Determine if dataset should be loaded from hub or locally."},
    )
    cleaned_files_dir_path: Optional[str] = field(
        default=os.getenv("CLEANED_FILES_DIR_PATH"),
        metadata={"help": "Path to the directory for cleaned files"},
    )

    ### DO TOKENIZER TRAINING ###
    do_train_tokenizer: bool = field(default=False)
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})
    block_size: int = field(default=512, metadata={"help": "Define the block size."})
    clean_text: bool = field(default=True)
    handle_chinese_chars: bool = field(default=False)
    strip_accents: bool = field(default=False)
    lowercase: bool = field(default=True)
    vocab_size: int = 30522
    limit_alphabet: int = 1000
    min_frequency: int = 2

    do_push_tokenizer_to_hub: bool = field(
        default=False, metadata={"help": "Enable or disable pushing tokenizer to hub."}
    )

    ### REFORMAT FILES ###
    do_reformat_files: bool = field(
        default=False, metadata={"help": "Enable or disable file reformatting."}
    )
    sliding_window_size: int = field(
        default=8, metadata={"help": "Size of the sliding window for processing data."}
    )

    ### SPLIT PATHS ###
    do_split_paths: bool = field(default=False)

    ### TOKENIZE FILES ###
    do_tokenize_files: bool = field(default=False)
    paths: str = field(
        default="train_test",
        metadata={"help": "Which file paths to use: all, train, test, or train_test."},
    )

    ### MASK TOKENS ###
    do_create_masked_encodings: bool = field(default=False)
    mlm_type: str = field(
        default="manual",
        metadata={
            "help": "Type of masking to use for masked language modeling. Pass either 'manual' or 'automatic'"
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )

    ### TRAIN MODEL ###
    do_train_model: bool = field(default=False)
    trainer_type: str = field(
        default="pytorch",
        metadata={"help": "Type of trainer to use: pytorch or huggingface"},
    )
    seed: int = field(default=42, metadata={"help": "Seed for reproducibility"})
    hidden_size: int = field(default=768)
    num_attention_heads: int = field(default=12)
    num_hidden_layers: int = field(default=12)
    type_vocab_size: int = field(default=2)

    train_batch_size: int = field(default=32, metadata={"help": "Training batch size."})
    eval_batch_size: int = field(default=8, metadata={"help": "Evaluation batch size."})
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Learning rate for training."}
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Number of training epochs."}
    )
    num_eval_epochs: int = field(
        default=10, metadata={"help": "Number of evaluation epochs."}
    )

    ### INFERENCING ###
    do_inference: bool = field(default=False)

    ### PUSHING DATA TO HUB ###
    do_login_first_time: bool = field(
        default=False,
        metadata={
            "help": "Toggle first-time login. Credentials will be cached after the initial login to the hub."
        },
    )
    huggingface_token: Optional[str] = field(default=os.getenv("HUGGINGFACE_TOKEN"))
    huggingface_repo_name: Optional[str] = field(
        default=os.getenv("HUGGINGFACE_REPO_NAME")
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    ### MERGING DATA ###
    if args.do_merge_docs:
        echo_with_color("Compiling the data into a single CSV file...", color="yellow")

        merge_docs(args.data_path)
    else:
        echo_with_color(
            "Skipping the data compilation into a single CSV file...",
            color="yellow",
        )

    ### CLEANING DATA ###
    if args.do_clean_data:
        echo_with_color("Cleaning the data...", color="red")

        clean_data(
            args.do_push_dataset_to_hub,
            args.do_login_first_time,
            args.huggingface_token,
            args.huggingface_repo_name,
        )
    else:
        echo_with_color("Skipping data cleaning.", color="red")

    ### FILE ANALYSIS ###
    if args.do_file_analysis:
        echo_with_color("Calculating file capacities...", color="green")
        calculate_file_capacities()

        echo_with_color("Comparing token counts...", color="green")
        compare_token_counts()
    else:
        echo_with_color("Skipping file analysis...", color="green")

    ### EXPORTING CSV TO TXT FILES ###
    if args.do_export_csv_to_txt_files:
        echo_with_color("Exporting CSV to txt files...", color="blue")

        export_csv_to_txt_files(
            args.cleaned_files_dir_path,
            args.do_load_dataset_from_hub,
            args.do_login_first_time,
            args.huggingface_token,
            args.huggingface_repo_name,
        )
    else:
        echo_with_color("Skipping export of CSV to txt files...", color="blue")

    ### TOKENIZER TRAINING ###
    if args.do_train_tokenizer:
        echo_with_color("Training a tokenizer from scratch...", color="black")

        train_tokenizer(
            args.model_type,
            args.cleaned_files_dir_path,
            args.block_size,
            args.clean_text,
            args.handle_chinese_chars,
            args.strip_accents,
            args.lowercase,
            args.vocab_size,
            args.limit_alphabet,
            args.min_frequency,
            args.do_push_tokenizer_to_hub,
            args.do_login_first_time,
            args.huggingface_token,
            args.huggingface_repo_name,
        )
    else:
        echo_with_color("Skipping the training of a tokenizer...", color="black")

    ### REFORMAT FILES ###
    if args.do_reformat_files:
        echo_with_color("Reformatting the files...", color="magenta")

        reformat_files(
            args.cleaned_files_dir_path,
            args.sliding_window_size,
        )
    else:
        echo_with_color("Skipping the reformatting of the files...", color="magenta")

    ### SPLIT PATHS ###
    if args.do_split_paths:
        echo_with_color(
            "Splitting the paths to train and test path sets...",
            color="white",
        )

        split_paths()
    else:
        echo_with_color(
            "Skipping the split of all paths to train and test path sets......",
            color="white",
        )

    ### TOKENIZE FILES ###
    if args.do_tokenize_files:
        echo_with_color(
            "Tokenizing the files and saving them...", color="bright_yellow"
        )

        tokenize_files(
            args.model_type,
            args.paths,
            args.block_size,
        )
    else:
        echo_with_color(
            "Skipping the tokenization of the files...",
            color="bright_yellow",
        )

    ### MASK TOKENS ###
    if args.do_create_masked_encodings:
        echo_with_color(
            "Creating masked encodings for the files...",
            color="bright_magenta",
        )

        create_masked_encodings(
            args.model_type,
            args.mlm_type,
            args.mlm_probability,
        )

    else:
        echo_with_color(
            "Skipping the creation of masked encodings...",
            color="bright_magenta",
        )

    ### TRAIN MODEL ###
    if args.do_train_model:
        echo_with_color("Training the model...", color="bright_cyan")

        train_model(
            model_type=args.model_type,
            trainer_type=args.trainer_type,
            seed=args.seed,
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            num_hidden_layers=args.num_hidden_layers,
            type_vocab_size=args.type_vocab_size,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            num_eval_epochs=args.num_eval_epochs,
            # do_login_first_time=args.do_login_first_time,
            # huggingface_token=args.huggingface_token,
            # huggingface_repo_name=args.huggingface_repo_name,
        )

    else:
        echo_with_color("Skipping the training of the model...", color="bright_cyan")

    ### INFERENCING ###
    if args.do_inference:
        echo_with_color("Inferencing...", color="bright_white")

        infer(
            model_type=args.model_type,
            model_path=None,
            block_size=args.block_size,
        )

    else:
        echo_with_color("Skipping inferencing...", color="bright_white")


if __name__ == "__main__":
    main()

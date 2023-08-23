import os
from dataclasses import dataclass, field
from typing import Optional

import typer
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
from src._05_data_tokenizing_and_masking.main import main as tokenize_files

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

    ### EXPORTING CSV TO TXT FILES ###
    do_export_csv_to_txt_files: bool = field(
        default=False,
        metadata={"help": "Enable or disable export of CSV to txt files."},
    )
    cleaned_files_dir_path: Optional[str] = field(
        default=os.getenv("CLEANED_FILES_DIR_PATH"),
        metadata={"help": "Path to the directory for cleaned files"},
    )

    ### DO FILE ANALYSIS ###
    do_file_analysis: bool = field(
        default=False, metadata={"help": "Enable or disable file analysis."}
    )

    ### DO TOKENIZER TRAINING ###
    do_train_tokenizer: bool = field(default=False)
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})
    block_size: int = field(default=512, metadata={"help": "Define the block size."})

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
    do_mask_tokens: bool = field(default=False)

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
    script_args = parser.parse_args_into_dataclasses()[0]

    ### MERGING DATA ###
    if script_args.do_merge_docs:
        typer.echo(
            typer.style(
                "Compiling the data into a single CSV file...", fg=typer.colors.YELLOW
            )
        )
        merge_docs(script_args.data_path)
    else:
        typer.echo(
            typer.style(
                "Skipping the data compilation into a single CSV file...",
                fg=typer.colors.YELLOW,
            )
        )

    ### CLEANING DATA ###
    if script_args.do_clean_data:
        typer.echo(typer.style("Cleaning the data...", fg=typer.colors.RED))

        clean_data(
            script_args.do_push_dataset_to_hub,
            script_args.do_login_first_time,
            script_args.huggingface_token,
            script_args.huggingface_repo_name,
        )
    else:
        typer.echo(typer.style("Skipping data cleaning.", fg=typer.colors.RED))

    if script_args.do_export_csv_to_txt_files:
        typer.echo(typer.style("Exporting CSV to txt files...", fg=typer.colors.BLUE))

        export_csv_to_txt_files(
            script_args.cleaned_files_dir_path,
            script_args.do_login_first_time,
            script_args.huggingface_token,
            script_args.huggingface_repo_name,
        )
    else:
        typer.echo(
            typer.style("Skipping export of CSV to txt files...", fg=typer.colors.BLUE)
        )

    ### FILE ANALYSIS ###
    if script_args.do_file_analysis:
        typer.echo(typer.style("Performing file analysis...", fg=typer.colors.GREEN))
        calculate_file_capacities()
        compare_token_counts()
    else:
        typer.echo(typer.style("Skipping file analysis...", fg=typer.colors.GREEN))

    ### TOKENIZER TRAINING ###
    if script_args.do_train_tokenizer:
        typer.echo(
            typer.style("Training a tokenizer from scratch...", fg=typer.colors.BLACK)
        )

        train_tokenizer(
            script_args.model_type,
            script_args.cleaned_files_dir_path,
            script_args.block_size,
            script_args.do_push_tokenizer_to_hub,
            script_args.do_login_first_time,
            script_args.huggingface_token,
            script_args.huggingface_repo_name,
        )
    else:
        typer.echo(
            typer.style(
                "Skipping the training of a tokenizer...", fg=typer.colors.BLACK
            )
        )

    ### REFORMAT FILES ###
    if script_args.do_reformat_files:
        typer.echo(typer.style("Reformatting the files...", fg=typer.colors.MAGENTA))

        reformat_files(
            script_args.cleaned_files_dir_path,
            script_args.sliding_window_size,
        )
    else:
        typer.echo(
            typer.style("Skipping file reformatting...", fg=typer.colors.MAGENTA)
        )

    if script_args.do_split_paths:
        typer.echo(
            typer.style(
                "Splitting all paths to train and test path sets...",
                fg=typer.colors.WHITE,
            )
        )

        split_paths()
    else:
        typer.echo(
            typer.style("Skipping the split of all paths......", fg=typer.colors.WHITE)
        )

    if script_args.do_tokenize_files:
        typer.echo(
            typer.style(
                "Tokenizing the files and saving them as TFRecords...",
                fg=typer.colors.BRIGHT_YELLOW,
            )
        )

        tokenize_files(
            script_args.model_type,
            script_args.paths,
            script_args.block_size,
        )
    else:
        typer.echo(
            typer.style(
                "Skipping the tokenization of the files...",
                fg=typer.colors.BRIGHT_YELLOW,
            )
        )


if __name__ == "__main__":
    main()

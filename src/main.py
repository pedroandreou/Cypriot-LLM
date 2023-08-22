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

load_dotenv(find_dotenv())


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

    ### EXPORTING CSV TO TXT FILES ###
    do_export_csv_to_txt_files: bool = field(
        default=False,
        metadata={"help": "Enable or disable export of CSV to txt files."},
    )
    output_dir_path: Optional[str] = field(
        default=os.getenv("CLEANED_FILES_DIR_PATH"),
        metadata={"help": "Path to the directory for cleaned files"},
    )

    ### DO FILE ANALYSIS ###
    do_file_analysis: bool = field(
        default=False, metadata={"help": "Enable or disable file analysis."}
    )

    ### PUSHING DATA TO HUB ###
    do_push_to_hub: bool = field(
        default=False, metadata={"help": "Enable or disable push to hub."}
    )
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
            script_args.do_push_to_hub,
            script_args.do_login_first_time,
            script_args.huggingface_token,
            script_args.huggingface_repo_name,
        )
    else:
        typer.echo(typer.style("Skipping data cleaning.", fg=typer.colors.RED))

    if script_args.do_export_csv_to_txt_files:
        typer.echo(typer.style("Exporting CSV to txt files...", fg=typer.colors.BLUE))

        export_csv_to_txt_files(
            script_args.output_dir_path,
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


if __name__ == "__main__":
    main()

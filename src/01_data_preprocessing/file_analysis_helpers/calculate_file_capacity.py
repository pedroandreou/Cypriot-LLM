import os
from dataclasses import dataclass, field

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.table import Table
from transformers import HfArgumentParser

load_dotenv(find_dotenv())
console = Console()


@dataclass
class ScriptArguments:
    input_file_name: str = field(
        default=f"../data_cleaner/preprocessed_docs.csv",
        metadata={"help": "Path to the input file"},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    df = pd.read_csv(script_args.input_file_name)

    # calculate the size of the content column in bytes
    total_size_in_bytes = df["content"].str.len().sum()
    total_size_in_kb = total_size_in_bytes / 1024
    total_size_in_mb = total_size_in_kb / 1024
    total_size_in_gb = total_size_in_mb / 1024

    table = Table()
    table.add_column("Filename", style="cyan", no_wrap=True)
    table.add_column("Bytes", style="red bold", no_wrap=True, min_width=12)
    table.add_column("Kilobytes (KB)", style="red bold", no_wrap=True, min_width=12)
    table.add_column("Megabytes (MB)", style="red bold", no_wrap=True, min_width=12)
    table.add_column("Gigabytes (GB)", style="red bold", no_wrap=True, min_width=12)

    table.add_row(
        script_args.input_file_name,
        str(total_size_in_bytes),
        str(total_size_in_kb),
        str(total_size_in_mb),
        str(total_size_in_gb),
    )
    console.print(table)


if __name__ == "__main__":
    main()

from dataclasses import dataclass, field
from typing import List

import pandas as pd
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.table import Table
from transformers import HfArgumentParser

"""
    Compares the token counts of multiple file paths provided as arguments.

    At least two file paths should be provided and it calculates the token counts for each file.
    A comparison is then made between the token counts of the files, and the results are returned.
"""


console = Console()


def count_tokens(df):
    token_count = 0

    for _, row in df.iterrows():
        content = row["content"]
        tokens = word_tokenize(content)
        token_count += len(tokens)

    return token_count


@dataclass
class ScriptArguments:
    files: List[str] = field(
        default_factory=list, metadata={"help": "List of file paths"}
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if len(script_args.files) < 2:
        print("At least two file paths must be provided for comparison.")
        exit()

    table = Table("Filename", "Token count")
    for file in script_args.files:
        df = pd.read_csv(f"{file}.csv")
        # Calculate token count for df
        token_count_df = count_tokens(df)

        table.add_row(file, str(token_count_df))

    console.print(table)


if __name__ == "__main__":
    main()

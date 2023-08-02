from typing import List

import pandas as pd
import typer
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.table import Table

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


def main(files: List[str]):
    if len(files) < 2:
        typer.echo("At least two file paths must be provided for comparison.")
        raise typer.Exit()

    table = Table("Filename", "Token count")
    for file in files:
        df = pd.read_csv(file)
        # Calculate token count for df
        token_count_df = count_tokens(df)

        table.add_row(file, str(token_count_df))

    console.print(table)


if __name__ == "__main__":
    typer.run(main)

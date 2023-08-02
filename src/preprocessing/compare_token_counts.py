from typing import List

import pandas as pd
import typer
from nltk.tokenize import word_tokenize


"""
    Compares the token counts of multiple file paths provided as arguments.

    At least two file paths should be provided and it calculates the token counts for each file.
    A comparison is then made between the token counts of the files, and the results are returned.
"""


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

    for file in files:
        df = pd.read_csv(file)
        # Calculate token count for df
        token_count_df = count_tokens(df)
        print(f"Token count in {file}: {token_count_df}")


if __name__ == "__main__":
    typer.run(main)

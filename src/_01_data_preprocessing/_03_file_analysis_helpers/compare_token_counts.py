import os

import pandas as pd
from nltk.tokenize import word_tokenize
from rich import print
from rich.console import Console
from rich.table import Table

"""
    Compares the token counts of the initial and the preprocessed files.
    A comparison is then made between the token counts of the files, and the results are returned in a table.
"""


def count_tokens(df):
    token_count = 0

    for _, row in df.iterrows():
        content = row["content"]
        tokens = word_tokenize(content)
        token_count += len(tokens)

    return token_count


console = Console()
curr_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    initial_doc_file_path = os.path.normpath(
        os.path.join(curr_dir, "..", "_01_doc_merge_to_csv", "all_documents.csv")
    )
    preprocessed_doc_file_path = os.path.normpath(
        os.path.join(curr_dir, "..", "_02_data_cleaner", "preprocessed_docs.csv")
    )
    files = [initial_doc_file_path, preprocessed_doc_file_path]

    if len(files) < 2:
        print(
            f"[bold red]At least two file paths must be provided for comparison.[/bold red]"
        )
        exit()

    table = Table("Filename", "Token count")
    for file in files:
        df = pd.read_csv(file)
        # Calculate token count for df
        token_count_df = count_tokens(df)

        table.add_row(os.path.basename(file), str(token_count_df))

    console.print(table)


if __name__ == "__main__":
    main()

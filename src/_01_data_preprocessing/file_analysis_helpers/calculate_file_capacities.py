import os

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(find_dotenv())


console = Console()
script_directory = os.path.dirname(os.path.abspath(__file__))


def main():

    initial_doc_file_path = os.path.normpath(
        os.path.join("..", "doc_merge_to_csv", "all_documents.csv")
    )
    preprocessed_doc_file_path = os.path.normpath(
        os.path.join("..", "data_cleaner", "preprocessed_docs.csv")
    )

    filepaths = [initial_doc_file_path, preprocessed_doc_file_path]

    for filepath in filepaths:
        df = pd.read_csv(filepath)

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
            filepath,
            str(total_size_in_bytes),
            str(total_size_in_kb),
            str(total_size_in_mb),
            str(total_size_in_gb),
        )
        console.print(table)


if __name__ == "__main__":
    main()

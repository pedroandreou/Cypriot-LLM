import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

console = Console()


def main(input_file_name: str = "preprocessed_docs"):
    df = pd.read_csv(f"{input_file_name}.csv")

    # calculate the size of the content column in bytes
    total_size_in_bytes = df["content"].str.len().sum()
    total_size_in_kb = total_size_in_bytes / 1024
    total_size_in_mb = total_size_in_kb / 1024
    total_size_in_gb = total_size_in_mb / 1024

    print(f"Total size of contents in the CSV file:\n")

    table = Table()
    table.add_column("Filename", style="cyan", no_wrap=True)
    table.add_column("Bytes", style="red bold", no_wrap=True, min_width=12)
    table.add_column("Kilobytes (KB)", style="red bold", no_wrap=True, min_width=12)
    table.add_column("Megabytes (MB)", style="red bold", no_wrap=True, min_width=12)
    table.add_column("Gigabytes (GB)", style="red bold", no_wrap=True, min_width=12)

    table.add_row(
        input_file_name,
        str(total_size_in_bytes),
        str(total_size_in_kb),
        str(total_size_in_mb),
        str(total_size_in_gb),
    )
    console.print(table)


if __name__ == "__main__":
    typer.run(main)

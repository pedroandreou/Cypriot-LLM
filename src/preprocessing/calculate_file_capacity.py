import pandas as pd
import typer


def main(input_file_name: str = "preprocessed_docs"):
    df = pd.read_csv(f"{input_file_name}.csv")

    # calculate the size of the content column in bytes
    total_size_in_bytes = df["content"].str.len().sum()
    total_size_in_kb = total_size_in_bytes / 1024
    total_size_in_mb = total_size_in_kb / 1024
    total_size_in_gb = total_size_in_mb / 1024

    print(f"Total size of contents in the CSV file:\n")
    print(f"{total_size_in_bytes} bytes")
    print(f"{total_size_in_kb} kilobytes (KB)")
    print(f"{total_size_in_mb} megabytes (MB)")
    print(f"{total_size_in_gb} gigabytes (GB)")


if __name__ == "__main__":
    typer.run(main)

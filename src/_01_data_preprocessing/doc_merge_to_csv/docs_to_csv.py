import glob
import os

import docx
import pandas as pd
import textract
from dotenv import find_dotenv, load_dotenv
from pdfminer.high_level import extract_text
from rich import print

load_dotenv(find_dotenv())
import argparse

from tqdm import tqdm

"""
This script reads different document formats (PDF, Docx, ODT), extracts their text content, and creates a DataFrame from it.

Pre-Execution Notes:
1. Prior Manual Conversion:
    - Prior to running this script, all DOC files were manually converted to DOCX on the local machine. This was necessary for proper operation, as the script relies on specific libraries for parsing these document types.
    - Some DOCX files were also manually converted to PDF as they could not be opened otherwise.
2. TXT File Exclusion:
    - TXT files were intentionally not processed due to ethical considerations; web-scraped data.

These manual conversions were performed prior to the execution of this script and are mentioned here to provide context for the person running the script again in the future.
"""


class DocumentReader:
    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def read_pdf(path):
        text = extract_text(path)

        return text

    @staticmethod
    def read_docx(path):
        doc = docx.Document(path)
        text = ""
        for p in doc.paragraphs:
            text += p.text + "\n"

        return text

    @staticmethod
    def read_odt(path):
        return textract.process(path).decode("utf-8", errors="ignore")

    def create_dataframe(self):
        data = []

        extensions = ["pdf", "docx", "odt"]
        total_files = sum(
            1
            for ext in extensions
            for _ in glob.glob(
                os.path.join(self.data_path, "**", f"*.{ext}"), recursive=True
            )
        )

        # Initialize the tqdm progress bar with the total number of files
        pbar = tqdm(total=total_files, desc="Processing files")

        for ext in extensions:
            for path in glob.glob(
                os.path.join(self.data_path, "**", f"*.{ext}"), recursive=True
            ):
                try:
                    content = None

                    if ext == "pdf":
                        content = self.read_pdf(path)
                    elif ext == "docx":
                        content = self.read_docx(path)
                    elif ext == "odt":
                        content = self.read_odt(path)

                    # Append the extracted text
                    if content is not None:
                        filename = os.path.splitext(os.path.basename(path))[0]
                        data.append([filename, content])
                    else:
                        print(
                            f"[bold red]{path} was processed but no content was extracted[/bold red]"
                        )

                    # Update the progress bar by 1 for each file processed
                    pbar.update(1)

                except Exception as e:
                    print(f"[bold red]Could not process {path}: {e}[/bold red]")

        pbar.pandas(desc="Saving dataframe to CSV")
        pbar.close()

        df = pd.DataFrame(data, columns=["filename", "content"])

        return df


script_directory = os.path.dirname(os.path.abspath(__file__))


def main(data_path):
    reader = DocumentReader(data_path)

    # Create the dataframe
    df = reader.create_dataframe()
    print(df.head())

    output_file_name = "all_documents.csv"
    output_file = os.path.join(script_directory, output_file_name)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv("DATASET_DIR_PATH"),
        help="Path to the dataset directory.",
    )

    args = parser.parse_args()

    main(args.data_path)

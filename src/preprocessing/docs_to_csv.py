import glob
import os

import docx
import pandas as pd
import textract
import typer
from pdfminer.high_level import extract_text
from rich import print

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

        for ext in ["pdf", "docx", "odt"]:
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

                except Exception as e:
                    print(f"[bold red]Could not process {path}: {e}[/bold red]")

        df = pd.DataFrame(data, columns=["filename", "content"])

        return df


def main(
    data_path: str = "G:\\My Drive\\Uni\\Masters\\Thesis\\dataset",
    output_file_name: str = "all_documents",
):
    reader = DocumentReader(data_path)

    # Create the dataframe
    df = reader.create_dataframe()
    print(df.head())

    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_directory, f"{output_file_name}.csv")
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    typer.run(main)

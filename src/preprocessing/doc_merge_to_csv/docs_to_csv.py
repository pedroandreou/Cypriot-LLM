import glob
import os

import docx
import pandas as pd
import textract
import typer
from dotenv import find_dotenv, load_dotenv
from pdfminer.high_level import extract_text
from rich import print

from src.preprocessing.hub_pusher import push_dataset

load_dotenv(find_dotenv())

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


app = typer.Typer()


@app.command()
def main(
    merge_data: bool = typer.Option(
        False, help="Enable or disable data merging into a single CSV file."
    ),
    data_path: str = os.getenv("DATASET_DIR_PATH"),
    output_file_name: str = os.getenv("COMPILED_DOCS_FILE_NAME"),
    first_time_login: bool = typer.Option(
        False,
        help="Toggle first-time login. Credentials will be cached after the initial login to the hub.",
    ),
    push_to_hub: bool = typer.Option(False, help="Enable or disable push to hub."),
    huggingface_dataset_repo_name: str = os.getenv("HUGGINGFACE_DATASET_REPO_NAME"),
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN"),
):

    if merge_data:
        typer.echo("Compiling the data into a single CSV file...")

        reader = DocumentReader(data_path)

        # Create the dataframe
        df = reader.create_dataframe()
        print(df.head())

        script_directory = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_directory, output_file_name)
        df.to_csv(output_file, index=False)
    else:
        typer.echo("Skipping the data compilation into a single CSV file...")

    # Push all data to the hub
    if push_to_hub:
        push_dataset(
            huggingface_token,
            output_file_name,
            custom_key="all_data",
            first_time_login=first_time_login,
            huggingface_dataset_repo_name=huggingface_dataset_repo_name,
        )
    else:
        typer.echo("Skipping push to the hub...")


if __name__ == "__main__":
    typer.run(main)

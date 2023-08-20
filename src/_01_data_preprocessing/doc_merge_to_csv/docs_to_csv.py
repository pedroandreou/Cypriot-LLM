import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import docx
import pandas as pd
import textract
from dotenv import find_dotenv, load_dotenv
from pdfminer.high_level import extract_text
from rich import print
from transformers import HfArgumentParser

from src.hub_pusher import push_dataset

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


@dataclass
class ScriptArguments:
    merge_data: bool = field(
        default=False,
        metadata={"help": "Enable or disable data merging into a single CSV file."},
    )
    data_path: Optional[str] = field(
        default=os.getenv("DATASET_DIR_PATH"),
        metadata={"help": "Path to the dataset directory."},
    )
    output_file_name: Optional[str] = field(
        default="all_documents.csv",
        metadata={"help": "Name of the compiled output file."},
    )
    do_login_first_time: bool = field(
        default=False,
        metadata={
            "help": "Toggle first-time login. Credentials will be cached after the initial login to the hub."
        },
    )
    do_push_to_hub: bool = field(
        default=False, metadata={"help": "Enable or disable push to hub."}
    )
    huggingface_token: Optional[str] = field(
        default=os.getenv("HUGGINGFACE_TOKEN"),
        metadata={"help": "Hugging Face token for authentication."},
    )
    huggingface_repo_name: Optional[str] = field(
        default=os.getenv("HUGGINGFACE_REPO_NAME"),
        metadata={"help": "Name of the Hugging Face dataset repository."},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.merge_data:
        print("Compiling the data into a single CSV file...")

        reader = DocumentReader(script_args.data_path)

        # Create the dataframe
        df = reader.create_dataframe()
        print(df.head())

        script_directory = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_directory, script_args.output_file_name)
        df.to_csv(output_file, index=False)
    else:
        print("Skipping the data compilation into a single CSV file...")

    # Do not push this dataset to the Hub - leave the preprocessed to be pushed
    # More than one dataset in the same dataset repo is supported but you will find it difficult
    # to load the right dataset (preprocessed) when you want to export the csv docs to txt files
    if script_args.do_push_to_hub:
        print("Pushing dataset to the hub...")

        push_dataset(
            do_login_first_time=script_args.do_login_first_time,
            huggingface_token=script_args.huggingface_token,
            huggingface_repo_name=script_args.huggingface_repo_name,
            output_file_name=script_args.output_file_name,
            custom_key="all_data",
        )
    else:
        print("Skipping push to the hub...")


if __name__ == "__main__":
    main()

import glob
import os
from datetime import datetime

import docx
import pandas as pd
import textract
import typer
from PyPDF2 import PdfReader


"""
This script reads different document formats (PDF, Docx, ODT), extracts their text content, and creates a DataFrame from it.

Pre-Execution Notes:
1. Prior Manual Conversion:
    - Prior to running this script, all DOC files were manually converted to DOCX on the local machine. This was necessary for proper operation, as the script relies on specific libraries for parsing these document types.
    - Some DOCX files were also manually converted to PDF as they could not be opened otherwise.
2. TXT File Exclusion:
    - TXT files were intentionally not processed due to ethical considerations.

These manual conversions were performed prior to the execution of this script and are mentioned here to provide context for the person running the script again in the future. It's important to know that the data passed to this script had undergone these conversions manually.
"""


class DocumentReader:
    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def read_pdf(path):
        with open(path, "rb") as file:
            pdf_reader = PdfReader(file)

            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

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

        for ext in ["pdf", "docx", "odt"]:  # "txt"
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

                    filename = os.path.splitext(os.path.basename(path))[0]
                    data.append([filename, content])

                except Exception as e:
                    print(f"Could not process {path}: {e}")

        df = pd.DataFrame(data, columns=["filename", "content"])

        return df


if __name__ == "__main__":

    def main(data_path: str):
        reader = DocumentReader(data_path)

        df = reader.create_dataframe()
        print(df.head())

        script_directory = os.path.dirname(os.path.abspath(__file__))

        now = datetime.now()
        timestamp = now.strftime("%Y=%m=%d_%H=%M=%S")
        output_file = os.path.join(script_directory, f"document_data_{timestamp}.xlsx")

        df.to_excel(output_file, index=False)

    typer.run(main)

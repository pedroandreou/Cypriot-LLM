import glob
import os

import docx
import pandas as pd
import textract
import typer
from PyPDF2 import PdfReader


"""
Notes:
1.  Converted the doc file extensions to docx
2.  Ignored the txt files due to ethical reasons
"""


def read_pdf(path):
    with open(path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text


def read_docx(path):
    doc = docx.Document(path)
    text = ""
    for p in doc.paragraphs:
        text += p.text + "\n"

    return text


def read_odt(path):

    return textract.process(path).decode("utf-8", errors="ignore")


def create_dataframe(data_path: str):
    data = []

    for ext in ["pdf", "doc", "docx", "odt"]:  # "txt"
        for path in glob.glob(
            os.path.join(data_path, "**", f"*.{ext}"), recursive=True
        ):
            try:
                filename = os.path.splitext(os.path.basename(path))[0]

                if ext == "pdf":
                    content = read_pdf(path)
                elif ext == ["doc", "docx"]:
                    content = read_docx(path)
                elif (
                    ext in "odt"
                ):  # "txt" can be added here but won't be added due to ethical reasons
                    content = read_odt(path)

                data.append([filename, content])

            except Exception as e:
                print(f"Could not process {path}: {e}")

    df = pd.DataFrame(data, columns=["filename", "content"])
    return df


def main(data_path: str):
    df = create_dataframe(data_path)
    print(df.head())

    # Export DataFrame to Excel file in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_directory, "output.xlsx")
    df.to_excel(output_file, index=False)


if __name__ == "__main__":
    typer.run(main)

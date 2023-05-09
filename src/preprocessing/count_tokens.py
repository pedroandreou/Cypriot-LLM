import glob
import os

import docx
import textract
import typer
from PyPDF2 import PdfReader


def count_tokens_pdf(path):
    tokens = 0
    with open(path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            tokens += len(page.extract_text().split())
    return tokens


def count_tokens_docx(path):
    tokens = 0
    doc = docx.Document(path)
    for p in doc.paragraphs:
        tokens += len(p.text.split())

    return tokens


def count_tokens_txt_odt(path):
    try:
        text = textract.process(path).decode("utf-8")
    except UnicodeDecodeError:
        text = textract.process(path, encoding="utf-8", errors="ignore").decode("utf-8")

    return len(text.split())


def count_tokens(data_path: str):
    total_tokens = 0

    for ext in ["pdf", "docx", "odt"]:  # "txt"
        for path in glob.glob(
            os.path.join(data_path, "**", f"*.{ext}"), recursive=True
        ):
            try:
                if ext == "pdf":
                    tokens = count_tokens_pdf(path)
                elif ext == "docx":
                    tokens = count_tokens_docx(path)
                elif (
                    ext in "odt"
                ):  # "txt" can be added here but won't be added due to ethical reasons
                    tokens = count_tokens_txt_odt(path)
                total_tokens += tokens
            except Exception as e:
                print(f"Could not process {path}: {e}")

    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    typer.run(count_tokens)

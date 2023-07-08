import os
import shutil

import typer
from PyPDF2 import PdfReader, PdfWriter


def split_pdf(file: str):
    pdf = PdfReader(file)
    num_pages = len(pdf.pages)
    pages_per_file = 10  # number of pages per split file

    for i in range(0, num_pages, pages_per_file):
        pdf_writer = PdfWriter()

        for j in range(i, min(i + pages_per_file, num_pages)):
            pdf_writer.add_page(pdf.pages[j])

        output_dir = os.path.basename(file)

        # Check if directory exists
        if os.path.isdir(f"./{output_dir}"):
            shutil.rmtree(f"./{output_dir}")

        os.mkdir(f"./{output_dir}")
        output_filename = f"./new_docs/{output_dir}/{output_dir}_{i + 1}_to_{min(i + pages_per_file, num_pages)}.pdf"

        with open(output_filename, "wb") as out:
            pdf_writer.write(out)

        print(f"Created: {output_filename}")


def main():
    file = r"G:\My Drive\Uni\Masters\Thesis\dataset"
    split_pdf(file)


if __name__ == "__main__":
    typer.run(main)

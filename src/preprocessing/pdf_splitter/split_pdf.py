import os
import shutil

import typer
from PyPDF2 import PdfReader, PdfWriter


def main(
    input_file_name: str = r"G:\My Drive\Uni\Masters\Thesis\dataset\Miscellaneous\keimena_kypriakis_logotexnias_b.pdf",
):
    output_dir_name = os.path.basename(input_file_name)
    # Check if directory exists
    if os.path.isdir(f"./{output_dir_name}"):
        shutil.rmtree(f"./{output_dir_name}")
    os.mkdir(f"./{output_dir_name}")

    pdf = PdfReader(input_file_name)
    num_pages = len(pdf.pages)
    pages_per_file = 10  # number of pages per split file

    for i in range(0, num_pages, pages_per_file):
        pdf_writer = PdfWriter()

        for j in range(i, min(i + pages_per_file, num_pages)):
            pdf_writer.add_page(pdf.pages[j])

        output_filename = f"./{output_dir_name}/{output_dir_name}_{i + 1}_to_{min(i + pages_per_file, num_pages)}.pdf"
        with open(output_filename, "wb") as out:
            pdf_writer.write(out)

        print(f"Created: {output_filename}")


if __name__ == "__main__":
    typer.run(main)

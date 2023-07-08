import typer
from PyPDF2 import PdfReader, PdfWriter


def split_pdf(file: str):
    pdf = PdfReader(file)
    num_pages = len(pdf.pages)
    pages_per_file = 50  # number of pages per split file

    for i in range(0, num_pages, pages_per_file):
        pdf_writer = PdfWriter()

        for j in range(i, min(i + pages_per_file, num_pages)):
            pdf_writer.add_page(pdf.pages[j])

        output_filename = f"./new_docs/keimena_kypriakis_logotexnias_b_{i + 1}_to_{min(i + pages_per_file, num_pages)}.pdf"

        with open(output_filename, "wb") as out:
            pdf_writer.write(out)

        print(f"Created: {output_filename}")


def main():
    file = "G:\\My Drive\\Uni\\Masters\\Thesis\\dataset\\Miscellaneous\\keimena_kypriakis_logotexnias_b.pdf"
    split_pdf(file)


if __name__ == "__main__":
    typer.run(main)

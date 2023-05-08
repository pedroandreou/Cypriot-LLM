import glob

import typer
import docx


def count_tokens(data_path):
    tokens = 0
    for path in glob.glob(data_path + "/**/*.docx"):
        try:
            doc = docx.Document(path)
        except docx.opc.exceptions.PackageNotFoundError:
            print(f"Could not open {path}")

        for p in doc.paragraphs:
            tokens += len(p.text.split())

    print(f"Total tokens: {tokens}")


if __name__ == "__main__":
    typer.run(count_tokens)
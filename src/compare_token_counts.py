from typing import List

import pandas as pd
import typer
from nltk.tokenize import word_tokenize


def count_tokens(df):
    token_count = 0

    for _, row in df.iterrows():
        content = row["content"]
        tokens = word_tokenize(content)
        token_count += len(tokens)

    return token_count


def compare_content(files: List[str]):
    for file in files:
        df = pd.read_excel(file)
        # Calculate token count for df
        token_count_df = count_tokens(df)
        print(f"Token count in {file}: {token_count_df}")


def main():
    typer.run(compare_content)


if __name__ == "__main__":
    main()

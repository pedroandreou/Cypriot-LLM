import pandas as pd
from nltk.tokenize import word_tokenize


def compare_content(file1, file2, file3):
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    df3 = pd.read_excel(file3)

    # Calculate token count for df1
    token_count_df1 = count_tokens(df1)
    print(f"Token count in {file1}: {token_count_df1}")

    # Calculate token count for df2
    token_count_df2 = count_tokens(df2)
    print(f"Token count in {file2}: {token_count_df2}")

    # Calculate token count for df2
    token_count_df3 = count_tokens(df3)
    print(f"Token count in {file3}: {token_count_df3}")


def count_tokens(df):
    token_count = 0

    for _, row in df.iterrows():
        content = row["content"]
        tokens = word_tokenize(content)
        token_count += len(tokens)

    return token_count


# Example usage
file1 = "document_data_2023=06=14_14=51=41.xlsx"
file2 = "document_data_2023=06=14_14=55=12.xlsx"
file3 = "output.xlsx"
compare_content(file1, file2, file3)

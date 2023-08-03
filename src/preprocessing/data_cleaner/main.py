import pandas as pd
import typer
from diacritic_remover import DiacriticRemover
from greek_letter_joiner import GreekLetterJoiner
from pattern_remover import PatternRemover


def main(
    input_file_name: str = "all_documents", output_file_name: str = "preprocessed_docs"
):
    df = pd.read_csv(f"{input_file_name}.csv")

    # Apply the remove_caron_generic function on the content column
    df["content"] = df["content"].apply(
        lambda text: DiacriticRemover(text)
        .remove_caron()
        .remove_breve()
        .remove_low_acute()
        .remove_diaeresis()
        .text
    )

    # Remove the patterns
    df["content"] = df["content"].apply(
        lambda text: PatternRemover(text).remove_patterns()
    )

    # Join single letters and unite the single Greek lowercase vowel letter with the previous word
    # if its last character is a Greek lowercase single vowel
    # Example: "τ ζ α ι" -- > "τζαι"
    # "πρέπε ι" --> "πρέπει"
    df["content"] = df["content"].apply(
        lambda text: GreekLetterJoiner(text)
        .reverse_text()
        .handle_single_letters()
        .reverse_text()
        .handle_uppercase()
        .handle_oti()
        .combine_vowels()
        .text
    )

    # Save preprocessed dataframe as a new csv file
    df.to_csv(f"{output_file_name}.csv", index=False)


if __name__ == "__main__":
    typer.run(main)

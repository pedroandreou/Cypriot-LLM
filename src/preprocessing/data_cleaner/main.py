import os

import pandas as pd
import typer
from diacritic_remover import DiacriticRemover
from dotenv import find_dotenv, load_dotenv
from greek_letter_joiner import GreekLetterJoiner
from pattern_remover import PatternRemover

from src.hub_pusher import push_dataset

load_dotenv(find_dotenv())


app = typer.Typer()


@app.command()
def main(
    clean_data: bool = typer.Option(False, help="Enable or disable data cleaning."),
    input_file_name: str = typer.Option(
        f"../doc_merge_to_csv/{os.getenv('COMPILED_DOCS_FILE_NAME')}",
        help="Path to the compiled documents CSV file.",
    ),
    output_file_name: str = typer.Option(
        os.getenv("PREPROCESSED_DOCS_FILE_NAME"),
        help="Name of the file to save the preprocessed documents CSV file.",
    ),
    push_to_hub: bool = typer.Option(False, help="Enable or disable push to hub."),
    first_time_login: bool = typer.Option(
        False,
        help="Toggle first-time login. Credentials will be cached after the initial login to the hub.",
    ),
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN"),
    huggingface_dataset_repo_name: str = os.getenv("HUGGINGFACE_DATASET_REPO_NAME"),
):
    if clean_data:
        typer.echo("Cleaning the data...")

        df = pd.read_csv(input_file_name)

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
        df.to_csv(output_file_name, index=False)

    else:
        typer.echo("Skipping data cleaning...")

    # Push the preprocessed data to the hub
    if push_to_hub:
        push_dataset(
            first_time_login=first_time_login,
            huggingface_token=huggingface_token,
            huggingface_dataset_repo_name=huggingface_dataset_repo_name,
            output_file_name=output_file_name,
            custom_key="preprocessed_data",
        )
    else:
        typer.echo("Skipping push to the hub...")


if __name__ == "__main__":
    typer.run(main)

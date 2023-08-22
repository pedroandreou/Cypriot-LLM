import argparse
import os

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.hub_pusher import push_dataset

from .diacritic_remover import DiacriticRemover
from .greek_letter_joiner import GreekLetterJoiner
from .pattern_remover import PatternRemover

load_dotenv(find_dotenv())


script_directory = os.path.dirname(os.path.abspath(__file__))


def main(
    do_push_to_hub,
    do_login_first_time,
    huggingface_token,
    huggingface_repo_name,
):

    input_file_name = os.path.normpath(
        os.path.join(script_directory, "..", "doc_merge_to_csv", "all_documents.csv")
    )
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
    output_file_name = "preprocessed_docs.csv"
    df.to_csv(output_file_name, index=False)

    # Push the preprocessed data to the hub
    if do_push_to_hub:
        print("Pushing dataset to the hub...")

        push_dataset(
            do_login_first_time=do_login_first_time,
            huggingface_token=huggingface_token,
            huggingface_repo_name=huggingface_repo_name,
            output_file_name=output_file_name,
            custom_key="preprocessed_data",
        )
    else:
        print("Skipping push to the hub...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument(
        "--do_push_to_hub",
        type=bool,
        default=False,
        help="Enable or disable push to hub.",
    )
    parser.add_argument(
        "--do_login_first_time",
        type=bool,
        default=False,
        help="Toggle first-time login.",
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=os.getenv("HUGGINGFACE_TOKEN"),
        help="Hugging Face token for authentication.",
    )
    parser.add_argument(
        "--huggingface_repo_name",
        type=str,
        default=os.getenv("HUGGINGFACE_REPO_NAME"),
        help="Name of the Hugging Face dataset repository.",
    )

    args = parser.parse_args()

    main(
        args.do_push_to_hub,
        args.do_login_first_time,
        args.huggingface_token,
        args.huggingface_repo_name,
    )

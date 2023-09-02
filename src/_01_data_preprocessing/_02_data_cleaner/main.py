import os

import pandas as pd

from src.utils.common_utils import echo_with_color
from src.utils.hub_pusher import push_dataset

from .diacritic_remover import DiacriticRemover
from .greek_letter_joiner import GreekLetterJoiner
from .pattern_remover import PatternRemover

curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(
    do_push_dataset_to_hub,
):

    input_file_name = os.path.normpath(
        os.path.join(curr_dir, "..", "_01_doc_merge_to_csv", "all_documents.csv")
    )

    try:
        df = pd.read_csv(input_file_name)
    except FileNotFoundError:
        raise Exception(
            "The compiled CSV file of all docs does not exist. You should run the doc_merge_to_csv script first."
        )

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
    output_file = os.path.join(curr_dir, output_file_name)
    df.to_csv(output_file, index=False)

    # Push the preprocessed data to the hub
    if do_push_dataset_to_hub:

        echo_with_color(
            "Pushing dataset to the hub...",
            color="red",
        )

        push_dataset(
            output_file_name=output_file_name,
            custom_key="preprocessed_data",
        )
    else:
        echo_with_color(
            "Skipping push to the hub...",
            color="red",
        )


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Dataset push and login arguments
    parser.add_argument(
        "--do_push_dataset_to_hub",
        type=bool,
        default=False,
        help="Enable or disable push to hub. Accepts: True/False. Default is False.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_arguments()

    main(
        args.do_push_dataset_to_hub,
    )

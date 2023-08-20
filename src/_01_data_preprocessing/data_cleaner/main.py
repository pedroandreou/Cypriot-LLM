import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from diacritic_remover import DiacriticRemover
from dotenv import find_dotenv, load_dotenv
from greek_letter_joiner import GreekLetterJoiner
from pattern_remover import PatternRemover
from transformers import HfArgumentParser

from src.hub_pusher import push_dataset

load_dotenv(find_dotenv())


@dataclass
class ScriptArguments:
    do_clean_data: bool = field(
        default=False, metadata={"help": "Enable or disable data cleaning."}
    )
    input_file_name: Optional[str] = field(
        default="../doc_merge_to_csv/all_documents.csv",
        metadata={"help": "Path to the compiled documents CSV file."},
    )
    output_file_name: Optional[str] = field(
        default="preprocessed_docs.csv",
        metadata={
            "help": "Name of the file to save the preprocessed documents CSV file."
        },
    )
    do_push_to_hub: bool = field(
        default=False, metadata={"help": "Enable or disable push to hub."}
    )
    do_login_first_time: bool = field(
        default=False,
        metadata={
            "help": "Toggle first-time login. Credentials will be cached after the initial login to the hub."
        },
    )
    huggingface_token: Optional[str] = field(default=os.getenv("HUGGINGFACE_TOKEN"))
    huggingface_repo_name: Optional[str] = field(
        default=os.getenv("HUGGINGFACE_REPO_NAME")
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.do_clean_data:
        print("Cleaning the data...")

        df = pd.read_csv(script_args.input_file_name)

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
        df.to_csv(script_args.output_file_name, index=False)

    else:
        print("Skipping data cleaning...")

    # Push the preprocessed data to the hub
    if script_args.do_push_to_hub:
        print("Pushing dataset to the hub...")

        push_dataset(
            do_login_first_time=script_args.do_login_first_time,
            huggingface_token=script_args.huggingface_token,
            huggingface_repo_name=script_args.huggingface_repo_name,
            output_file_name=script_args.output_file_name,
            custom_key="preprocessed_data",
        )
    else:
        print("Skipping push to the hub...")


if __name__ == "__main__":
    main()

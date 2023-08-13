import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from transformers import HfArgumentParser

from src.hub_pusher import hub_login

load_dotenv(find_dotenv())


@dataclass
class ScriptArguments:
    output_dir_path: Optional[str] = field(
        default=os.getenv("CLEANED_FILES_DIR_PATH"),
        metadata={"help": "Path to the directory for cleaned files"},
    )
    first_time_login: bool = field(
        default=False,
        metadata={
            "help": "Toggle first-time login. Credentials will be cached after the initial login to the hub."
        },
    )
    huggingface_token: Optional[str] = field(default=os.getenv("HUGGINGFACE_TOKEN"))
    huggingface_dataset_repo_name: Optional[str] = field(
        default=os.getenv("HUGGINGFACE_DATASET_REPO_NAME")
    )
    custom_key: str = field(
        default="preprocessed_data",
        metadata={"help": "Custom key for preprocessed data"},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    hub_login(script_args.huggingface_token, script_args.first_time_login)

    # Load the dataset
    dataset = load_dataset(script_args.huggingface_dataset_repo_name)

    # Access the dataset
    dataset = dataset[script_args.custom_key]

    # Convert to Pandas DataFrame
    df = dataset.to_pandas()

    for num in range(len(df)):
        value = df.iloc[num, 1]

        with open(
            f"{script_args.output_dir_path}\\text_file{num}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(str(value))


if __name__ == "__main__":
    main()

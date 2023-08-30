import argparse
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from utils.hub_pusher import hub_login

curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(
    output_dir_path,
    do_load_dataset_from_hub,
    do_login_first_time,
    huggingface_token,
    huggingface_repo_name,
):

    if do_load_dataset_from_hub:
        hub_login(huggingface_token, do_login_first_time)

        # Load the dataset
        dataset = load_dataset(huggingface_repo_name)

        # Access the dataset
        custom_key = "preprocessed_data"
        dataset = dataset[custom_key]

        # Convert to Pandas DataFrame
        df = dataset.to_pandas()
    else:
        # Load it from locally
        df = pd.read_csv(
            os.path.join(curr_dir, "data_cleaner", "preprocessed_docs.csv")
        )

    for num in tqdm(range(len(df)), desc="Exporting CSV docs to TXT files"):
        value = df.iloc[num, 1]

        file_path = os.path.join(output_dir_path, f"text_file{num}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(value))


def parse_dataset_arguments():
    parser = argparse.ArgumentParser(description="Script parameters.")

    parser.add_argument(
        "--output_dir_path",
        type=str,
        default=os.getenv("CLEANED_FILES_DIR_PATH"),
        help="Path to the output directory for cleaned files.",
    )

    parser.add_argument(
        "--do_load_dataset_from_hub",
        type=bool,
        default=False,
        help="Determine if dataset should be loaded from hub. Accepts: True/False. Default is False, indicating the dataset should be loaded locally.",
    )

    parser.add_argument(
        "--do_login_first_time",
        type=bool,
        default=False,
        help="Toggle first-time login. Accepts: True/False",
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

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    args = parse_dataset_arguments()

    main(
        args.output_dir_path,
        args.do_load_dataset_from_hub,
        args.do_login_first_time,
        args.huggingface_token,
        args.huggingface_repo_name,
    )

import argparse
import os

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from src.utils.hub_pusher import load_custom_dataset

load_dotenv(find_dotenv())


curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(
    output_dir_path,
    do_load_dataset_from_hub,
):

    if do_load_dataset_from_hub:
        dataset = load_custom_dataset("preprocessed_data")

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


def parse_arguments():
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

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_arguments()

    main(
        args.output_dir_path,
        args.do_load_dataset_from_hub,
    )

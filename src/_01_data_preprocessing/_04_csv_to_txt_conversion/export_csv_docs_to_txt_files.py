import os

import pandas as pd
from tqdm import tqdm

from utils.hub_pusher import load_custom_dataset

curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(
    do_load_dataset_from_hub: bool,
):

    if do_load_dataset_from_hub:
        dataset = load_custom_dataset("preprocessed_data")

        df = dataset.to_pandas()
    else:
        # Load it from locally
        df = pd.read_csv(
            os.path.join(curr_dir, "..", "_02_data_cleaner", "preprocessed_docs.csv")
        )

    for num in tqdm(range(len(df)), desc="Exporting CSV docs to TXT files"):
        value = df.iloc[num, 1]

        file_path = os.path.join(curr_dir, "cleaned_files", f"text_file{num}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(value))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script parameters.")

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
        args.do_load_dataset_from_hub,
    )

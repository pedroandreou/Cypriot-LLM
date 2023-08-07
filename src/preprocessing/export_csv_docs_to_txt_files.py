import os

import typer
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def main(
    output_dir_path: str = os.getenv("CLEANED_FILES_DIR_PATH"),
    huggingface_dataset_repo_name: str = os.getenv("HUGGINGFACE_DATASET_REPO_NAME"),
):
    # Load the dataset
    dataset = load_dataset(huggingface_dataset_repo_name)

    # Access the dataset
    dataset = dataset["preprocessed_data"]

    # Convert to Pandas DataFrame
    df = dataset.to_pandas()

    for num in range(len(df)):
        value = df.iloc[num, 1]

        with open(f"{output_dir_path}\\text_file{num}.txt", "w", encoding="utf-8") as f:
            f.write(str(value))


if __name__ == "__main__":
    typer.run(main)

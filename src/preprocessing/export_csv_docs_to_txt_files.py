import os

import typer
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv

from src.hub_pusher import hub_login

load_dotenv(find_dotenv())


app = typer.Typer()


@app.command()
def main(
    output_dir_path: str = os.getenv("CLEANED_FILES_DIR_PATH"),
    first_time_login: bool = typer.Option(
        False,
        help="Toggle first-time login. Credentials will be cached after the initial login to the hub.",
    ),
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN"),
    huggingface_dataset_repo_name: str = os.getenv("HUGGINGFACE_DATASET_REPO_NAME"),
    custom_key: str = "preprocessed_data",
):
    hub_login(huggingface_token, first_time_login)

    # Load the dataset
    dataset = load_dataset(huggingface_dataset_repo_name)

    # Access the dataset
    dataset = dataset[custom_key]

    # Convert to Pandas DataFrame
    df = dataset.to_pandas()

    for num in range(len(df)):
        value = df.iloc[num, 1]

        with open(f"{output_dir_path}\\text_file{num}.txt", "w", encoding="utf-8") as f:
            f.write(str(value))


if __name__ == "__main__":
    typer.run(main)

import pandas as pd
import typer
from datasets import Dataset, DatasetDict
from huggingface_hub import login


def push_dataset(
    huggingface_token: str,
    output_file_name: str,
    custom_key: str,
    first_time_login: bool = False,
    huggingface_dataset_repo_name: str = "your_repo_name",
):
    typer.echo("Pushing to the hub...")

    if first_time_login:
        typer.echo("Logging in...")
        login(token=huggingface_token)
    else:
        typer.echo("Skipping logging in since credentials are in the cache...")

    # Load the dataset
    df = pd.read_csv(output_file_name)

    # Convert DataFrame to a Dataset
    dataset = Dataset.from_pandas(df)

    # Create a DatasetDict with a custom key
    custom_dataset_dict = DatasetDict({custom_key: dataset})

    # Push to HuggingFace Hub
    custom_dataset_dict.push_to_hub(huggingface_dataset_repo_name, private=True)

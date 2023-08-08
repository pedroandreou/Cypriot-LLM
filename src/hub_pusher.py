import pandas as pd
import typer
from datasets import Dataset, DatasetDict
from huggingface_hub import login


def hub_login(token: str, first_time_login: bool) -> None:
    if first_time_login:
        typer.echo("Logging in...")
        login(token=token)
    else:
        typer.echo("Skipping logging in since credentials are in the cache...")


def push_dataset(
    first_time_login: bool,
    huggingface_token: str,
    huggingface_dataset_repo_name: str,
    output_file_name: str,
    custom_key: str,
) -> None:
    typer.echo("Pushing dataset to the hub...")
    hub_login(huggingface_token, first_time_login)

    # Load the dataset and convert DataFrame to a Dataset
    dataset = Dataset.from_pandas(pd.read_csv(output_file_name))

    # Create a DatasetDict with a custom key and push to HuggingFace Hub
    DatasetDict({custom_key: dataset}).push_to_hub(
        huggingface_dataset_repo_name, private=True
    )


def push_tokenizer(
    tokenizer,
    first_time_login: bool,
    huggingface_token: str,
    huggingface_repo_name: str,
) -> None:
    typer.echo("Pushing tokenizer to the hub...")
    hub_login(huggingface_token, first_time_login)

    # Push tokenizer to HuggingFace Hub
    tokenizer.push_to_hub(huggingface_repo_name, private=True)

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login


def hub_login(token: str, do_login_first_time: bool) -> None:
    if do_login_first_time:
        print("Logging in...")
        login(token=token)
    else:
        print("Skipping logging in since credentials are in the cache...")


def push_dataset(
    do_login_first_time: bool,
    huggingface_token: str,
    huggingface_dataset_repo_name: str,
    output_file_name: str,
    custom_key: str,
) -> None:
    print("Pushing dataset to the hub...")
    hub_login(huggingface_token, do_login_first_time)

    # Load the dataset and convert DataFrame to a Dataset
    dataset = Dataset.from_pandas(pd.read_csv(output_file_name))

    # Create a DatasetDict with a custom key and push to HuggingFace Hub
    DatasetDict({custom_key: dataset}).push_to_hub(
        huggingface_dataset_repo_name, private=True
    )


def push_tokenizer(
    tokenizer,
    do_login_first_time: bool,
    huggingface_token: str,
    huggingface_repo_name: str,
) -> None:
    hub_login(huggingface_token, do_login_first_time)

    # Push tokenizer to HuggingFace Hub
    tokenizer.push_to_hub(huggingface_repo_name, private=True)

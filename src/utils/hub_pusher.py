import os
import shutil
import subprocess

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login

huggingface_repo_name = os.getenv("HUGGINGFACE_REPO_NAME")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
do_login_first_time = False


def hub_login() -> None:
    if do_login_first_time:
        print("Logging in...")
        login(token=huggingface_token)
    else:
        print("Skipping logging in since credentials are in the cache...")


def push_dataset(
    do_login_first_time: bool,
    huggingface_token: str,
    huggingface_repo_name: str,
    output_file_name: str,
    custom_key: str,
) -> None:
    hub_login(huggingface_token, do_login_first_time)

    # Load the dataset and convert DataFrame to a Dataset
    dataset = Dataset.from_pandas(pd.read_csv(output_file_name))

    # Create a DatasetDict with a custom key and push to HuggingFace Hub
    DatasetDict({custom_key: dataset}).push_to_hub(huggingface_repo_name, private=True)


def load_pushed_dataset(custom_key: str):
    hub_login(huggingface_token, do_login_first_time)

    # Load the dataset
    dataset = load_dataset(huggingface_repo_name)

    # Access the dataset
    custom_key = "preprocessed_data"
    dataset = dataset[custom_key]

    return dataset


def push_tokenizer(
    curr_dir: str,
    tokenizer_paths: list,
) -> None:
    hub_login(huggingface_token, do_login_first_time)

    # Step 1: Clone the repository
    repository_url = f"https://huggingface.co/{huggingface_repo_name}"

    os.chdir(os.path.join(curr_dir, "huggingface"))
    subprocess.run(["git", "clone", repository_url], check=True)
    repository_name = huggingface_repo_name.split("/")[
        -1
    ]  # Exclude the username and get only the repo name

    # Step 2: Move the tokenizer files
    for path in tokenizer_paths:
        filename = os.path.basename(path)  # Get the filename from the path
        shutil.move(path, os.path.join(repository_name, filename))

    # Step 3: Push the tokenizer to Hugging Face
    os.chdir(repository_name)

    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Add tokenizer"], check=True)
    subprocess.run(["git", "push"], check=True)

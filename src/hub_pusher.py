import os
import shutil
import subprocess

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
    huggingface_repo_name: str,
    output_file_name: str,
    custom_key: str,
) -> None:
    hub_login(huggingface_token, do_login_first_time)

    # Load the dataset and convert DataFrame to a Dataset
    dataset = Dataset.from_pandas(pd.read_csv(output_file_name))

    # Create a DatasetDict with a custom key and push to HuggingFace Hub
    DatasetDict({custom_key: dataset}).push_to_hub(huggingface_repo_name, private=True)


def push_tokenizer(
    curr_dir: str,
    tokenizer_paths: list,
    do_login_first_time: bool,
    huggingface_token: str,
    huggingface_repo_name: str,
) -> None:
    hub_login(huggingface_token, do_login_first_time)

    # Step 1: Clone the repository
    repository_url = f"https://huggingface.co/{huggingface_repo_name}"

    os.chdir(f"{curr_dir}/huggingface")
    subprocess.run(["git", "clone", repository_url], check=True)
    repository_name = huggingface_repo_name.split("/")[
        -1
    ]  # Exclude the username and get only the repo name

    # Step 2: Move the tokenizer files
    # Assuming vocab.txt is in the current directory
    for path in tokenizer_paths:
        shutil.move(path, f"{repository_name}/vocab.txt")

    # Step 3: Push the tokenizer to Hugging Face
    os.chdir(repository_name)

    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Add tokenizer"], check=True)
    subprocess.run(["git", "push"], check=True)

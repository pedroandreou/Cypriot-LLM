import argparse
import os

from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv

from src.hub_pusher import hub_login

load_dotenv(find_dotenv())


def main(
    output_dir_path,
    do_login_first_time,
    huggingface_token,
    huggingface_repo_name,
):

    hub_login(huggingface_token, do_login_first_time)

    # Load the dataset
    dataset = load_dataset(huggingface_repo_name)

    # Access the dataset
    custom_key = "preprocessed_data"
    dataset = dataset[custom_key]

    # Convert to Pandas DataFrame
    df = dataset.to_pandas()

    for num in range(len(df)):
        value = df.iloc[num, 1]

    file_path = os.path.join(output_dir_path, f"text_file{num}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument(
        "--output_dir_path",
        type=str,
        default=os.getenv("CLEANED_FILES_DIR_PATH"),
        help="Path to the output directory for cleaned files.",
    )
    parser.add_argument(
        "--do_login_first_time",
        type=bool,
        default=False,
        help="Toggle first-time login.",
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

    args = parser.parse_args()

    main(
        args.output_dir_path,
        args.do_login_first_time,
        args.huggingface_token,
        args.huggingface_repo_name,
    )

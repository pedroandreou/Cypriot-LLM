import os

from src._04_path_splitting.main import PathSplitter
from src._05_data_tokenizing.tokenized_dataset import TokenizedDataset
from src.utils.common_utils import (
    echo_with_color,
    get_new_subdirectory_path,
    save_dataset,
)

curr_dir = os.path.dirname(os.path.abspath(__file__))


def tokenize_and_save_dataset(
    model_type, tokenizer_version, paths, block_size, key, out_path
):
    echo_with_color(f"Tokenizing {key} files", color="bright_yellow")
    tokenized_dataset = TokenizedDataset(
        model_type=model_type,
        tokenizer_version=tokenizer_version,
        files_list=paths,
        block_size=block_size,
    )

    echo_with_color(f"Saving the tokenized {key} dataset...", color="bright_yellow")
    save_dataset(out_path, "tokenized", key, tokenized_dataset)


def main(model_type, tokenizer_version, block_size):

    train_paths, test_paths = PathSplitter.load_paths()

    # Create a directory for the tokenized dataset
    dataset_dir_path_w_model_type = os.path.join(
        curr_dir, "encodings", f"cy{model_type}"
    )
    dataset_dir_path_w_model_type_n_version = get_new_subdirectory_path(
        dataset_dir_path_w_model_type, "encodings"
    )

    echo_with_color("Tokenizing files", color="bright_yellow")
    tokenize_and_save_dataset(
        model_type,
        tokenizer_version,
        train_paths,
        block_size,
        "train",
        dataset_dir_path_w_model_type_n_version,
    )
    tokenize_and_save_dataset(
        model_type,
        tokenizer_version,
        test_paths,
        block_size,
        "test",
        dataset_dir_path_w_model_type_n_version,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script parameters.")

    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
    )

    parser.add_argument(
        "--tokenizer_version", type=int, default="1", help="Version of tokenizer to use"
    )

    parser.add_argument("--block_size", type=int, default=512, help="Block size.")

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_arguments()

    main(
        model_type=args.model_type,
        tokenizer_version=args.tokenizer_version,
        block_size=args.block_size,
    )

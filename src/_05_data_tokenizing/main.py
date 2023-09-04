import os

from src._04_path_splitting.main import PathSplitter
from src._05_data_tokenizing.tokenized_dataset import TokenizedDataset
from src.utils.common_utils import echo_with_color, save_dataset

curr_dir = os.path.dirname(os.path.abspath(__file__))


def tokenize_and_save_dataset(key, paths, model_type, tokenizer_version, block_size):
    echo_with_color(f"Tokenizing {key} files", color="bright_yellow")
    tokenized_dataset = TokenizedDataset(
        model_type=model_type,
        tokenizer_version=tokenizer_version,
        files_list=paths,
        block_size=block_size,
    )
    echo_with_color(f"Saving the tokenized {key} dataset...", color="bright_yellow")
    save_dataset(
        curr_dir, tokenized_dataset, f"encodings/cy{model_type}", "tokenized", key
    )


def main(model_type, tokenizer_version, block_size):

    train_paths, test_paths = PathSplitter.load_paths()

    echo_with_color("Tokenizing files", color="bright_yellow")
    tokenize_and_save_dataset(
        "train", train_paths, model_type, tokenizer_version, block_size
    )
    tokenize_and_save_dataset(
        "test", test_paths, model_type, tokenizer_version, block_size
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

import os

import torch
import typer

# from masked_dataset import MaskedDataset
from src._05_data_tokenizing_and_masking.tokenized_dataset import LineByLineTextDataset


def fetch_txt_files(paths_type):
    paths_directory = os.path.normpath(
        os.path.join(curr_dir, "..", "_04_path_splitting", "file_paths")
    )

    file_mapping = {
        "all": "all_paths.txt",
        "train": "train_paths.txt",
        "test": "test_paths.txt",
        "train_test": ["train_paths.txt", "test_paths.txt"],
    }

    # Using a dictionary to store the path_type as the key and its corresponding paths as the value
    txt_files_dict = {}

    file_name = file_mapping.get(paths_type)

    if isinstance(file_name, list):
        for fn in file_name:
            key = fn.split("_")[0]  # Using the filename's prefix as the key
            print(f"Loading {key} paths")

            with open(os.path.join(paths_directory, fn), "r") as f:
                txt_files_dict[key] = f.read().splitlines()
    else:
        print(f"Loading {paths_type} paths")

        with open(os.path.join(paths_directory, file_name), "r") as f:
            txt_files_dict[paths_type] = f.read().splitlines()

    return txt_files_dict


def save_dataset(dataset, base_path, sub_dir, key):
    filename = os.path.join(curr_dir, base_path, f"{sub_dir}_{key}_dataset.pth")
    torch.save(dataset, filename)


curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(model_type, paths, block_size):

    tokenizer_dir_path = os.path.normpath(
        os.path.join(
            curr_dir,
            "..",
            "_02_tokenizer_training",
            "trained_tokenizer_bundle",
            f"cy{model_type}",
        )
    )

    # key is train or test
    # value is a list of paths
    files_list_dict = fetch_txt_files(paths)
    for key, files_list in files_list_dict.items():
        typer.echo(
            typer.style(f"Tokenizing {key} files", fg=typer.colors.BRIGHT_YELLOW)
        )

        tokenized_dataset = LineByLineTextDataset(
            model_type=model_type,
            tokenizer_dir_path=tokenizer_dir_path,
            files_list=files_list,
            block_size=block_size,
        )

        typer.echo(
            typer.style(
                f"Saving the tokenized {key} dataset...", fg=typer.colors.BRIGHT_YELLOW
            )
        )
        save_dataset(tokenized_dataset, "encodings", "tokenized", key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your script's description")

    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
    )

    parser.add_argument(
        "--paths",
        type=str,
        choices=["all", "train", "test", "train_test"],
        default="train_test",
        help="Which file paths to use: all, train, test, or train_test.",
    )

    parser.add_argument("--block_size", type=int, default=512, help="Block size.")

    # parser.add_argument(
    #     "--mlm_type",
    #     type=str,
    #     choices=["manual", "automatic"],
    #     default="manual",
    #     help="Type of masking to use for masked language modeling. Pass either 'manual' or 'automatic'"
    # )

    # parser.add_argument(
    #     "--mlm_probability",
    #     type=float,
    #     default=0.15,
    #     help="Ratio of tokens to mask for masked language modeling loss"
    # )

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        paths=args.paths,
        block_size=args.block_size,
    )

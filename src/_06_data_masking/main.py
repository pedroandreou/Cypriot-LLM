import os

from src._05_data_tokenizing.tokenized_dataset import TokenizedDataset
from src._06_data_masking.masked_dataset import MaskedDataset
from src.utils.common_utils import (
    echo_with_color,
    get_new_subdirectory_path,
    save_dataset,
)

curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(
    model_type: str, encodings_version: int, mlm_type: str, mlm_probability: float
):
    # Create a directory for the masked datasets
    masked_dataset_dir_path_w_model_type = os.path.join(
        curr_dir, "masked_encodings", f"cy{model_type}"
    )
    masked_dataset_dir_path_w_model_type_n_version = get_new_subdirectory_path(
        masked_dataset_dir_path_w_model_type, "masked_encodings"
    )

    echo_with_color("Loading the tokenized datasets...", color="bright_magenta")
    train_dataset, test_dataset = TokenizedDataset().load_encodings(
        model_type, encodings_version
    )

    echo_with_color("Creating masked datasets...", color="bright_magenta")
    masked_train_dataset = MaskedDataset(
        train_dataset,
        model_type,
        mlm_type,
        mlm_probability,
    )
    masked_test_dataset = MaskedDataset(
        test_dataset,
        model_type,
        mlm_type,
        mlm_probability,
    )

    echo_with_color("Saving masked datasets...", color="bright_magenta")
    save_dataset(
        masked_dataset_dir_path_w_model_type_n_version,
        "masked",
        "train",
        masked_train_dataset,
    )
    save_dataset(
        masked_dataset_dir_path_w_model_type_n_version,
        "masked",
        "test",
        masked_test_dataset,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for model setup and data handling."
    )

    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
    )

    parser.add_argument(
        "--encodings_version",
        type=int,
        default="1",
        help="Version of encodings to use",
    )

    parser.add_argument(
        "--mlm_type",
        type=str,
        choices=["manual", "automatic"],
        default="manual",
        help="Type of masking to use for masked language modeling. Pass either 'manual' or 'automatic'.",
    )

    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_arguments()

    main(
        model_type=args.model_type,
        encodings_version=args.encodings_version,
        mlm_type=args.mlm_type,
        mlm_probability=args.mlm_probability,
    )

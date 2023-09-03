import os

from src._05_data_tokenizing.tokenized_dataset import TokenizedDataset
from src._06_data_masking.masked_dataset import MaskedDataset
from src.utils.common_utils import echo_with_color, save_dataset

curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(model_type, mlm_type, mlm_probability):

    echo_with_color("Loading the tokenized datasets...", color="bright_magenta")
    train_dataset, test_dataset = TokenizedDataset().load_encodings(model_type)

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
        curr_dir,
        masked_train_dataset,
        f"masked_encodings/cy{model_type}",
        "masked",
        "train",
    )
    save_dataset(
        curr_dir,
        masked_test_dataset,
        f"masked_encodings/cy{model_type}",
        "masked",
        "test",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for model setup and data handling."
    )

    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
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
        mlm_type=args.mlm_type,
        mlm_probability=args.mlm_probability,
    )

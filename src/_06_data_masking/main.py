import os

from src._05_data_tokenizing.tokenized_dataset import LineByLineTextDataset
from src._06_data_masking.masked_dataset import MaskedDataset
from src.utils.common_utils import echo_with_color, save_dataset

curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(model_type, mlm_type, mlm_probability):

    echo_with_color("Loading the tokenized datasets...", color="bright_magenta")
    train_dataset, test_dataset = LineByLineTextDataset().load_encodings()

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
    save_dataset(curr_dir, masked_train_dataset, "masked_encodings", "masked", "train")
    save_dataset(curr_dir, masked_test_dataset, "masked_encodings", "masked", "test")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your script's description")

    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
    )

    parser.add_argument(
        "--mlm_type",
        type=str,
        choices=["manual", "automatic"],
        default="manual",
        help="Type of masking to use for masked language modeling. Pass either 'manual' or 'automatic'",
    )

    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    args = parser.parse_args()

    main(
        model_type=args.model_type,
        mlm_type=args.mlm_type,
        mlm_probability=args.mlm_probability,
    )

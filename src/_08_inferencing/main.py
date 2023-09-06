import json

from transformers import BertTokenizer, pipeline

from src._02_tokenizer_training.main import TokenizerWrapper
from src._07_model_training.main import load_model
from src.utils.common_utils import echo_with_color
import os


class PipelineWrapper:
    def __init__(self, model, tokenizer):
        self.fill = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    def predict_next_token(self, input_string):
        mask_token = self.fill.tokenizer.mask_token
        predictions = self.fill(f"{input_string} {mask_token}")

        for prediction in predictions:
            echo_with_color(json.dumps(prediction, indent=4), color="bright_white")

    def predict_specific_token_within_a_passing_sequence(self, examples):
        for example in examples:
            for prediction in self.fill(example):
                echo_with_color(
                    f"{prediction['sequence']}, confidence: {prediction['score']}",
                    color="bright_white",
                )
            echo_with_color("=" * 50)


def main(
    model_type: str,
    model_version: str,
    tokenizer_version: int,
    input_unmasked_sequence: str,
    # input_masked_sequences: list,
):

    echo_with_color("Loading the saved model...", color="bright_white")

    loaded_model = load_model(
        model_type,
        model_version,
    )

    echo_with_color("Loading the saved tokenizer...", color="bright_white")
    # Our trained tokenizer is not compatible with the pipeline API (as our tokenizer is from the 'tokenizers' while the API from 'transformers').
    # Therefore, we need to get the paths and load it from the 'transformers' library.
    config_path, vocab_path = TokenizerWrapper().get_tokenizer_paths(
        model_type,
        tokenizer_version,
    )

    # Load configurations
    with open(config_path, "r") as file:
        loaded_tokenizer_config = json.load(file)

    loaded_transformers_tokenizer = BertTokenizer.from_pretrained(
        os.path.dirname(
            vocab_path
        ),  # Using directory path due to deprecation of direct file path in transformers v5
        **loaded_tokenizer_config,
    )

    pipeline_wrapper = PipelineWrapper(loaded_model, loaded_transformers_tokenizer)
    pipeline_wrapper.predict_next_token(input_unmasked_sequence)
    # pipeline_wrapper.predict_specific_token_within_a_passing_sequence(
    #     input_masked_sequences
    # )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for setting up a pipeline for inferencing."
    )

    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
    )
    parser.add_argument(
        "--model_version", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--tokenizer_version", type=int, default=1, help="Version of tokenizer to use"
    )
    parser.add_argument(
        "--input_unmasked_sequence",
        type=str,
        default="είσαι",
        help="Define input sequence for its next token to be predicted.",
    )
    # parser.add_argument(
    #     "--input_masked_sequences",
    #     nargs="+",
    #     default="Θώρει τη [MASK]."
    #     "Η τηλεόραση, το [MASK], τα φώτα."
    #     "Μεν τον [MASK] κόρη μου.",
    #     help="Define list of input masked sequences to predict their masked tokens.",
    # )

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_arguments()

    main(
        args.model_type,
        args.model_version,
        args.tokenizer_version,
        args.input_unmasked_sequence,
        args.input_masked_sequences,
    )

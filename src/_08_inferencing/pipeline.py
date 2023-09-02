import json

from transformers import BertForMaskedLM, RobertaForMaskedLM, pipeline

from src._02_tokenizer_training.main import TokenizerWrapper
from src.utils.common_utils import echo_with_color


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


def main(model_type: str, model_path: str, block_size: int):

    echo_with_color("Loading the saved model...", color="bright_white")
    if model_type == "bert":
        loaded_model = BertForMaskedLM.from_pretrained(model_path)
    else:
        loaded_model = RobertaForMaskedLM.from_pretrained(model_path)

    echo_with_color("Loading the saved tokenizer...", color="bright_white")
    loaded_tokenizer = TokenizerWrapper().load_tokenizer(
        model_type,
        block_size,
    )

    pipeline_wrapper = PipelineWrapper(loaded_model, loaded_tokenizer)

    # method 1
    pipeline_wrapper.predict_next_token("είσαι")

    # method 2
    examples = [
        "Today's most trending hashtags on [MASK] is Donald Trump",
        "The [MASK] was cloudy yesterday, but today it's rainy.",
    ]
    pipeline_wrapper.predict_specific_token_within_a_passing_sequence(examples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )  # NEED TO CHANGE THIS
    parser.add_argument(
        "--block_size", type=int, default=512, help="Define the block size."
    )

    args = parser.parse_args()

    main(args.model_type, args.model_path, args.block_size)

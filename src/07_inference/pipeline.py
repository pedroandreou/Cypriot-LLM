import json

import typer
from transformers import BertForMaskedLM, RobertaForMaskedLM, pipeline


class PipelineWrapper:
    def __init__(self, model, tokenizer):

        typer.echo("Loading the saved model...")
        if model_type == "bert":
            loaded_model = BertForMaskedLM.from_pretrained(model_path)

        else:  # roberta
            loaded_model = RobertaForMaskedLM.from_pretrained(model_path)

        pipeline_wrapper = PipelineWrapper(loaded_model, loaded_tokenizer)

        # method 1
        pipeline_wrapper.predict_next_token("είσαι")

        # method 2
        examples = [
            "Today's most trending hashtags on [MASK] is Donald Trump",
            "The [MASK] was cloudy yesterday, but today it's rainy.",
        ]
        pipeline_wrapper.predict_specific_token_within_a_passing_sequence(examples)
        self.fill = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    def predict_next_token(self, input_string):
        mask_token = self.fill.tokenizer.mask_token
        predictions = self.fill(f"{input_string} {mask_token}")

        for prediction in predictions:
            print(json.dumps(prediction, indent=4))

    def predict_specific_token_within_a_passing_sequence(self, examples):
        for example in examples:
            for prediction in self.fill(example):
                print(f"{prediction['sequence']}, confidence: {prediction['score']}")
            print("=" * 50)

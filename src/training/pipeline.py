import json

from transformers import pipeline


class PipelineWrapper:
    def __init__(self, model, tokenizer):
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

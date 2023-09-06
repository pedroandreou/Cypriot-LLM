import difflib
import json
import os
from typing import List
import re

from rich.console import Console
from rich.table import Table
from transformers import BertTokenizer, pipeline

from src._02_tokenizer_training.main import TokenizerWrapper
from src._07_model_training.main import load_model
from src.utils.common_utils import echo_with_color


class PipelineWrapper:
    def __init__(self, model, tokenizer):
        self.fill = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        # self.console = Console(force_terminal=True)

    def _create_prediction_table(self, columns):
        table = Table(show_header=True, header_style="bold magenta")
        for column_name, column_attrs in columns:
            table.add_column(
                column_name,
                style=column_attrs.get("style", None),
                width=column_attrs.get("width", None),
            )
        return table

    def predict_next_token(self, input_string):
        mask_token = self.fill.tokenizer.mask_token
        predictions = self.fill(f"{input_string} {mask_token}")

        columns = [
            ("Score", {"style": "dim", "width": 12}),
            ("Token", {"style": "dim", "width": 12}),
            ("Token String", {"width": 20}),
            ("Sequence", {"width": 50}),
        ]

        print("\n\nThe input sequence is: ", input_string)

        table = self._create_prediction_table(columns)
        for prediction in predictions:
            table.add_row(
                str(prediction["score"]),
                str(prediction["token"]),
                prediction["token_str"],
                prediction["sequence"],
            )
        self.console.print(table)

    def predict_specific_token_within_a_passing_sequence(self, examples):
        columns = [
            ("Sequence", {"width": 50}),
            ("Score", {"style": "dim", "width": 12}),
        ]

        for example in examples:

            highlighted_example = example.replace("[MASK]", "[yellow][MASK][/yellow]")
            
            console = Console(force_terminal=True)
            console.print(f"\n\nThe input sequence is: {highlighted_example}")

            table = self._create_prediction_table(columns)

            for prediction in self.fill(example):
                diff_result = self._highlight_difference(
                    example, prediction["sequence"]
                )
                table.add_row(diff_result, str(prediction["score"]))
            
            console.print(table)

    @staticmethod
    def _highlight_difference(text1, text2):
        words1 = text1.split()
        words2 = text2.split()
        
        wordslen1 = len(words1)
        wordslen2 = len(words2)

        if wordslen1 != wordslen2:
            if wordslen1 > wordslen2:
                # Look if wordpiece was added to an existing word token
                extra_wordpiece_found = False

                # Remove the MASK token from text1
                pattern = r'\[MASK\][,.!?;:]*'
                text1 = re.sub(pattern, '', text1).strip()
                words1 = text1.split()


                highlighted_words = []
                
                for w1, w2 in zip(words1, words2):
                    if len(w1) == len(w2):
                        highlighted_words.append(w2)
                    else:
                        diff = len(w2) - len(w1)
                        common_part = w1
                        added_part = w2[len(w1):]  # Take the extra part of w2
                        highlighted_words.append(f"{common_part}[yellow]{added_part}[/yellow]")
                        extra_wordpiece_found = True


                if extra_wordpiece_found:
                    return ' '.join(highlighted_words)
                
                else:
                    # it means the mask token was jsut replaced by a whitespace
                    # in the prediction
                    return text2
            
            elif wordslen1 < wordslen2:
                mask_index = words1.index("[MASK]")

                diff = wordslen2 - wordslen1

                # Color the tokens in words2 based on the difference.
                for i in range(diff):
                    if mask_index + i < len(words2):
                        words2[mask_index + i] = f"[yellow]{words2[mask_index + i]}[/yellow]"
                
                return ' '.join(words2)
        else: 
            # equal lengths 
            # just yellow the token that replaced the mask token
            highlighted_words = []
            
            for w1, w2 in zip(words1, words2):
                if w1.strip(',.!') == '[MASK]' and w1 != w2:
                    highlighted_words.append(f"[yellow]{w2}[/yellow]")
                else:
                    highlighted_words.append(w2)
            
            return ' '.join(highlighted_words)


def main(
    model_type: str,
    model_version: str,
    tokenizer_version: int,
    input_unmasked_sequence: str,
    input_masked_sequences: List[str],
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
    # print("Predict next token based on the passing sequence")
    # pipeline_wrapper.predict_next_token(input_unmasked_sequence)
    
    print("\n\nPredict specific token within a passing a sequence")
    pipeline_wrapper.predict_specific_token_within_a_passing_sequence(
        input_masked_sequences
    )


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
    parser.add_argument(
        "--input_masked_sequences",
        nargs="+",
        default="Θώρει τη [MASK]."
        "Η τηλεόραση, το [MASK], τα φώτα."
        "Μεν τον [MASK] κόρη μου.",
        help="Define list of input masked sequences to predict their masked tokens.",
    )

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

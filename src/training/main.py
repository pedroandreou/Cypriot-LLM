import os

import typer
from dataset import TestTextDataset, TrainTextDataset
from model import ModelWrapper
from pipeline import PipelineWrapper
from tokenizer import TokenizerWrapper


def validate_path(path: str):
    if not os.path.exists(path):
        typer.echo(f"The path {path} does not exist.")
        raise typer.Exit()


def validate_model_type(model_type: str):
    valid_model_types = ["bert", "roberta"]
    if model_type not in valid_model_types:
        typer.echo(f"Model type should be one of {valid_model_types}")
        raise typer.Exit()


def validate_mlm_method(mlm_method: str):
    valid_mlm_methods = ["manual", "automatic"]
    if mlm_method not in valid_mlm_methods:
        typer.echo(f"MLM method should be one of {valid_mlm_methods}")
        raise typer.Exit()


def main(
    base_path: str = "/content/drive/MyDrive/Uni/Masters/Thesis",
    model_type: str = "bert",
    vocab_size: int = 30522,
    max_length: int = 512,
    mlm_method: str = "manual",
):

    validate_path(base_path)
    validate_model_type(model_type)
    validate_mlm_method(mlm_method)

    # Initialize tokenizer
    tokenizer_wrapper = TokenizerWrapper(
        base_path=base_path,
        model_type=model_type,
        vocab_size=vocab_size,
        max_length=max_length,
    )
    # Get tokenizer
    tokenizer = tokenizer_wrapper.get_tokenizer()
    # Get the list of file paths for training and testing datasets
    train_file_paths = tokenizer_wrapper.get_paths()[0]
    test_file_paths = tokenizer_wrapper.get_paths()[1]

    # Create an instance of TrainTextDataset for training data
    train_dataset = TrainTextDataset(
        tokenizer, train_file_paths, mlm_method=mlm_method, max_length=max_length
    )
    # Create an instance of TestTextDataset for testing data
    test_dataset = TestTextDataset(
        tokenizer, test_file_paths, mlm_method=mlm_method, max_length=max_length
    )

    # # Encoding example
    # print("The input ids of the encoded input is: ", text_dataset[2]['input_ids'][:10]) # show only the first 10 out of the 512
    # # Decoding example
    # print("The corresponding tokens of the input ids are:\n", text_dataset.decode_input_ids_to_string(text_dataset[2]['input_ids']))
    # # Check the vocab size
    # vocab = tokenizer_wrapper.get_tokenizer().get_vocab()
    # vocab_size = len(vocab)
    # print("\nThe vocab size is: ", vocab_size)

    ## Train model
    if mlm_method == "manual":
        model_wrapper = ModelWrapper(
            train_set=train_dataset,
            test_set=test_dataset,
            base_path=base_path,
            model_type=model_type,
            vocab_size=vocab_size,
            max_length=max_length,
        )
    else:  # automatic
        # Get the data collators of each set
        train_data_collator = train_dataset.get_data_collator()
        test_data_collator = test_dataset.get_data_collator()

        model_wrapper = ModelWrapper(
            train_set=train_dataset,
            test_set=test_dataset,
            data_collator=train_data_collator,
            base_path=base_path,
            model_type=model_type,
            vocab_size=vocab_size,
            max_length=max_length,
        )

    pipeline_wrapper = PipelineWrapper(
        model_wrapper.get_model(), tokenizer_wrapper.get_tokenizer()
    )
    # method 1
    result = pipeline_wrapper.predict_next_token("είσαι")
    # method 2
    examples = [
        "Today's most trending hashtags on [MASK] is Donald Trump",
        "The [MASK] was cloudy yesterday, but today it's rainy.",
    ]
    pipeline_wrapper.predict_specific_token_within_a_passing_sequence(examples)


if __name__ == "__main__":
    typer.run(main)

import os
import pickle
from pathlib import Path

import typer
from dataset import TestTextDataset, TrainTextDataset
from model import ModelWrapper
from pipeline import PipelineWrapper
from sklearn.model_selection import train_test_split
from tokenizer import TokenizerWrapper
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    RobertaForMaskedLM,
    RobertaTokenizer,
)


def validate_path(path: str):
    if not os.path.exists(path):
        typer.echo(f"The path {path} does not exist.")
        raise typer.Exit()


def validate_model_type(model_type: str):
    valid_model_types = ["bert", "roberta"]
    if model_type not in valid_model_types:
        typer.echo(f"Model type should be one of {valid_model_types}")
        raise typer.Exit()


def create_directory_if_does_not_exist(path: str):
    # Create the directory if not already there
    if not os.path.exists(path):
        os.makedirs(path)


def create_dataset(tokenizer, paths, DatasetClass, max_length):
    dataset = DatasetClass(tokenizer, paths, max_length=max_length)
    return dataset


def save_dataset(dataset, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)


def load_dataset(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"No dataset found at {file_path}")

    with open(file_path, "rb") as f:
        return pickle.load(f)


def main(
    base_path: str = "/content/drive/MyDrive/Uni/Masters/Thesis",
    model_type: str = "bert",
    vocab_size: int = 30522,
    max_length: int = 512,
    should_train_tokenizer: bool = False,
    should_create_train_test_sets: bool = False,
    should_train_model: bool = False,
    should_inference: bool = False,
    # train_batch_size: int = 64,
    # train_steps_per_epoch: int = 64,
    # validation_batch_size: int = 64,
    # validation_steps_per_epoch: int = 64,
    # epochs: int = 1,
    # freeze_bert_layer: bool = False,
    # learning_rate: float = 0.01,
    # seed: int = 42,
    # run_validation: bool = False,
    # model_dir: str = os.environ['SM_MODEL_DIR'],
    # train_data: str = os.environ['SM_CHANNEL_TRAIN'],
    # validation_data: str = os.environ['SM_CHANNEL_VALIDATION'],
    # output_dir: str = os.environ['SM_OUTPUT_DIR'],
    # num_gpus: int = int(os.environ['SM_NUM_GPUS']),
):

    validate_path(base_path)
    validate_model_type(model_type)

    # Set paths
    dir_path = f"{base_path}/Project/cy{model_type}"
    tokenizer_path = f"{dir_path}/tokenizer"

    if should_train_tokenizer:
        # Create dirs if do not exist
        create_directory_if_does_not_exist(dir_path)
        create_directory_if_does_not_exist(tokenizer_path)

        # Build tokenizer
        TokenizerWrapper(
            model_type=model_type,
            vocab_size=vocab_size,
            max_length=max_length,
        )

    # Load the tokenizer
    loaded_tokenizer = None
    if model_type == "bert":
        loaded_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, max_length)

    else:  # roberta
        loaded_tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, max_length)

    # Define save paths
    train_save_path = Path("train_dataset.pkl")
    test_save_path = Path("test_dataset.pkl")

    if should_create_train_test_sets:
        all_paths = [str(x) for x in Path(f"{base_path}/cleaned_files/").glob("*.txt")]

        # Get the list of file paths for training and testing datasets
        train_file_paths, test_file_paths = train_test_split(all_paths, test_size=0.2)

        # Create train set
        train_dataset = create_dataset(
            loaded_tokenizer, train_file_paths, TrainTextDataset, max_length
        )
        save_dataset(train_dataset, train_save_path)

        # Create test set
        test_dataset = create_dataset(
            loaded_tokenizer, test_file_paths, TestTextDataset, max_length
        )
        save_dataset(test_dataset, test_save_path)

    # Load datasets
    train_dataset = load_dataset(train_save_path)
    test_dataset = load_dataset(test_save_path)

    # e.g. "cybert" or "cyroberta"
    model_dir_path = f"{base_path}/Project/cy{model_type}/model"
    # "cybert/pytorch" or "cybert/huggingface"
    # "cyroberta/pytorch" or "cyroberta/huggingface"
    model_path = (
        f"{model_dir_path}/{('pytorch' if model_type == 'bert' else 'huggingface')}"
    )

    ## Train model
    model = None
    if should_train_model:
        create_directory_if_does_not_exist(model_dir_path)
        create_directory_if_does_not_exist(model_path)

        if model_type == "bert":
            model = ModelWrapper(
                train_set=train_dataset,
                test_set=test_dataset,
                model_type=model_type,
                vocab_size=vocab_size,
                max_length=max_length,
            )

        else:  # roberta
            # Cannot get them like that - need to find a different way
            train_data_collator = train_dataset.get_data_collator()
            test_data_collator = test_dataset.get_data_collator()

            model = ModelWrapper(
                train_set=train_dataset,
                test_set=test_dataset,
                data_collator=train_data_collator,
                model_type=model_type,
                vocab_size=vocab_size,
                max_length=max_length,
            )

        model.save_pretrained(model_path)

    # Load model
    loaded_model = None
    if model_type == "bert":
        loaded_model = BertForMaskedLM.from_pretrained(model_path)

    else:  # roberta
        loaded_model = RobertaForMaskedLM.from_pretrained(model_path)

    if should_inference:
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
    typer.run(main)

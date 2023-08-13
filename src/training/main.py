import os
import pickle
from pathlib import Path

import typer
from dotenv import find_dotenv, load_dotenv
from mlm_components.dataset import TestTextDataset, TrainTextDataset
from model_components.model import ModelWrapper
from tokenizer_components.path_splitter import PathSplitter
from tokenizer_components.tokenizer import TokenizerWrapper
from transformers import (
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    RobertaTokenizer,
)

from src.hub_pusher import push_tokenizer

load_dotenv(find_dotenv())


def validate_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        typer.echo(f"The directory {path} has just been created.")


def create_masked_encodings(tokenizer, paths, DatasetClass, max_length):
    dataset = DatasetClass(tokenizer, paths, max_length=max_length)
    return dataset


def save_masked_encodings(dataset, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)


def load_masked_encodings(file_path):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"No dataset found at {file_path}")

    with open(file_path, "rb") as f:
        return pickle.load(f)


def main(
    ### main dir ###
    cybert_dir_path: str = os.getenv("MAIN_DIR_PATH"),
    ### model type ###
    model_type: str = "bert",
    ### tokenizer ###
    do_split_paths: bool = False,
    do_train_tokenizer: bool = False,
    tokenizer_dir_path: str = os.getenv("TOKENIZER_DIR_PATH"),
    do_push_tokenizer_to_hub: bool = typer.Option(
        False, help="Enable or disable pushing tokenizer to hub."
    ),
    vocab_size: int = 30522,
    max_length: int = 512,
    ### model ###
    do_create_train_test_sets: bool = False,  # Create train and test sets
    do_train_model: bool = False,  # Train modells
    train_ecodings_file_path: str = os.getenv("TRAIN_DATASET_ENCODINGS_FILE_PATH"),
    test_encodings_file_path: str = os.getenv("TEST_DATASET_ENCODINGS_FILE_PATH"),
    # train_batch_size: int = 64,
    # train_steps_per_epoch: int = 64,
    # validation_batch_size: int = 64,
    # validation_steps_per_epoch: int = 64,
    # epochs: int = 1,
    # freeze_bert_layer: bool = False,
    # learning_rate: float = 0.01,
    # seed: int = 42,
    # run_validation: bool = False,
    ## huggingface ###
    first_time_login: bool = typer.Option(
        False,
        help="Toggle first-time login. Credentials will be cached after the initial login to the hub.",
    ),
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN"),
    huggingface_dataset_repo_name: str = os.getenv("HUGGINGFACE_DATASET_REPO_NAME"),
):
    # Create the main dir path if does not exist
    validate_path(cybert_dir_path)

    if do_split_paths:
        typer.echo("Splitting all paths to train and test path sets...")

        path_splitter = PathSplitter()
        path_splitter.split_paths()
        path_splitter.save_paths()

    else:
        typer.echo(
            "Skipping the split of all paths...\nWill try to load the train and test path sets from the files."
        )

        path_splitter = PathSplitter()
        path_splitter.load_paths()

    # Get paths
    all_paths_list, train_paths_list, test_paths_list = path_splitter.get_paths()

    if do_train_tokenizer:
        typer.echo("Training a tokenizer from scratch...")

        TokenizerWrapper(
            filepaths=all_paths_list,
            tokenizer_path=tokenizer_dir_path,
            model_type=model_type,
            vocab_size=vocab_size,
            max_length=max_length,
        )

    else:
        typer.echo("Skipping the training of a tokenizer from scratch...")

    typer.echo("Loading the saved tokenizer...")
    if model_type == "bert":
        loaded_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir_path)

    else:  # roberta
        loaded_tokenizer = RobertaTokenizer.from_pretrained(tokenizer_dir_path)

    # Push tokenizer to Hub
    if do_push_tokenizer_to_hub:
        push_tokenizer(
            tokenizer=loaded_tokenizer,
            first_time_login=first_time_login,
            huggingface_token=huggingface_token,
            huggingface_repo_name=huggingface_dataset_repo_name,
        )
    else:
        typer.echo("Skipping the push of the tokenizer to the hub...")

    if do_create_train_test_sets:

        typer.echo("Creating masked encodings of the train set...")
        train_dataset = create_masked_encodings(
            loaded_tokenizer, train_paths_list, TrainTextDataset, max_length
        )

        typer.echo("Saving the masked encodings of the train set...")
        save_masked_encodings(train_dataset, train_ecodings_file_path)

        typer.echo("Creating masked encodings of the test set...")
        test_dataset = create_masked_encodings(
            loaded_tokenizer, test_paths_list, TestTextDataset, max_length
        )

        typer.echo("Saving the masked encodings of the test set...")
        save_masked_encodings(test_dataset, test_encodings_file_path)

    # Train model
    if do_train_model:
        typer.echo("Loading the masked encodings of the train and test sets...")
        train_dataset = load_masked_encodings(train_ecodings_file_path)
        test_dataset = load_masked_encodings(test_encodings_file_path)

        if model_type == "bert":
            typer.echo("Training a BERT model using the PyTorch API...")

            ModelWrapper(
                train_set=train_dataset,
                test_set=test_dataset,
                model_type=model_type,
                vocab_size=vocab_size,
                max_length=max_length,
            )

        else:  # roberta
            typer.echo("Training a RoBeRTa model using the HuggingFace API...")

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=loaded_tokenizer, mlm=True, mlm_probability=0.15
            )

            ModelWrapper(
                train_set=train_dataset,
                test_set=test_dataset,
                data_collator=data_collator,
                model_type=model_type,
                vocab_size=vocab_size,
                max_length=max_length,
            )

    else:
        typer.echo("Skipping the training of a model from scratch...")


if __name__ == "__main__":
    typer.run(main)

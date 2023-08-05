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
    DataCollatorForLanguageModeling,
)
from dotenv import load_dotenv

load_dotenv()


def validate_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        typer.echo(f"The directory {path} has just been created.")


def validate_model_type(model_type: str):
    valid_model_types = ["bert", "roberta"]
    if model_type not in valid_model_types:
        typer.echo(f"Model type should be one of {valid_model_types}")
        raise typer.Exit()


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
    cleaned_files_dir_path: str = os.getenv("CLEANED_FILES_DIR_PATH"),
    cybert_dir_path: str = os.getenv("CYBERT_DIR_PATH"),
    cyroberta_dir_path: str = os.getenv("CYROBERTA_DIR_PATH"),
    cybert_tokenizer_dir_path: str = os.getenv("CYBERT_TOKENIZER_DIR_PATH"),
    cyroberta_tokenizer_dir_path: str = os.getenv("CYROBERTA_TOKENIZER_DIR_PATH"),
    cybert_model_dir_path: str = os.getenv("CYBERT_MODEL_DIR_PATH"),
    cyroberta_model_dir_path: str = os.getenv("CYROBERTA_MODEL_DIR_PATH"),
    model_type: str = "bert",
    vocab_size: int = 30522,
    max_length: int = 512,
    should_train_tokenizer: bool = False,
    should_split_train_test: bool = False,
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
):

    validate_model_type(model_type)
    validate_path(cleaned_files_dir_path)
    if model_type == "bert":
        validate_path(cybert_dir_path)

        # Set paths
        tokenizer_path = cybert_tokenizer_dir_path
        model_path = cybert_model_dir_path

    else:  # roberta
        validate_path(cyroberta_dir_path)

        # Set paths
        tokenizer_path = cyroberta_tokenizer_dir_path
        model_path = cyroberta_model_dir_path
    validate_path(tokenizer_path)
    validate_path(model_path)

    if should_split_train_test:
        all_paths = [str(x) for x in Path(cleaned_files_dir_path).glob("*.txt")]
        # Get the list of file paths for training and testing datasets
        train_file_paths, test_file_paths = train_test_split(all_paths, test_size=0.2)

        # Saving the lists
        with open("training_testing_file_paths/train_file_paths.txt", "w") as f:
            for item in train_file_paths:
                f.write("%s\n" % item)
        with open("training_testing_file_paths/test_file_paths.txt", "w") as f:
            for item in test_file_paths:
                f.write("%s\n" % item)
    else:
        try:
            # Loading the lists
            with open("training_testing_file_paths/train_file_paths.txt", "r") as f:
                train_file_paths = f.read().splitlines()
                print("the train file paths are: ", train_file_paths)

            with open("training_testing_file_paths/test_file_paths.txt", "r") as f:
                test_file_paths = f.read().splitlines()
                print("the test file paths are: ", test_file_paths)

        except FileNotFoundError:
            typer.echo(
                f"train_file_paths.txt and test_file_paths.txt not found.\nYou should run the script with --should-split-train-test flag first."
            )
            raise typer.Exit()

    # Train/Build a tokenizer
    if should_train_tokenizer:
        TokenizerWrapper(
            train_paths=train_file_paths,
            tokenizer_path=tokenizer_path,
            model_type=model_type,
            vocab_size=vocab_size,
            max_length=max_length,
        )

    # Load the tokenizer
    if model_type == "bert":
        loaded_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    else:  # roberta
        loaded_tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    # Define save paths
    train_save_path = Path("dataset_encodings/train_dataset.pkl")
    test_save_path = Path("dataset_encodings/test_dataset.pkl")
    if should_create_train_test_sets:
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

    # Train model
    if should_train_model:
        if model_type == "bert":
            ModelWrapper(
                train_set=train_dataset,
                test_set=test_dataset,
                model_type=model_type,
                vocab_size=vocab_size,
                max_length=max_length,
            )

        else:  # roberta
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=loaded_tokenizer, mlm=True, mlm_probability=0.15
            )

            ModelWrapper(
                train_set=train_dataset,
                test_set=test_dataset,
                data_collator=data_collator,
                model_path=model_path,
                model_type=model_type,
                vocab_size=vocab_size,
                max_length=max_length,
            )

    # Load model
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

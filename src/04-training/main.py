import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from mlm_components.dataset import TestTextDataset, TrainTextDataset
from model_components.model import ModelWrapper
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    set_seed,
)

from src.hub_pusher import push_tokenizer

load_dotenv(find_dotenv())


def validate_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"The directory {path} has just been created.")


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


@dataclass
class ScriptArguments:
    main_dir_path: Optional[str] = field(
        default=os.getenv("MAIN_DIR_PATH"), metadata={"help": "Main directory path"}
    )
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})

    do_create_masked_encodings: bool = field(
        default=False, metadata={"help": "Create train and test sets"}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )

    do_train_model: bool = field(default=False, metadata={"help": "Train model"})
    train_encodings_file_path: Optional[str] = field(
        default=os.getenv("TRAIN_SET_ENCODINGS_FILE_PATH")
    )
    test_encodings_file_path: Optional[str] = field(
        default=os.getenv("TEST_DATASET_ENCODINGS_FILE_PATH")
    )
    learning_rate: float = field(
        default=0.01, metadata={"help": "Learning Rate for the training"}
    )
    max_steps: int = field(
        default=1_000_000, metadata={"help": "The Number of Training steps to perform"}
    )
    seed: int = field(default=42, metadata={"help": "Seed for reproducibility"})

    do_login_first_time: bool = field(
        default=False,
        metadata={
            "help": "Toggle first-time login. Credentials will be cached after the initial login to the hub."
        },
    )
    huggingface_token: Optional[str] = field(default=os.getenv("HUGGINGFACE_TOKEN"))
    huggingface_dataset_repo_name: Optional[str] = field(
        default=os.getenv("HUGGINGFACE_DATASET_REPO_NAME")
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Create the main dir path if does not exist
    validate_path(script_args.main_dir_path)

    # set seed for reproducibility
    set_seed(script_args.seed)

    print("Loading the saved tokenizer...")
    loaded_tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_dir_path)

    # Push tokenizer to Hub
    if script_args.do_push_tokenizer_to_hub:
        print("Pushing tokenizer to the hub...")

        push_tokenizer(
            tokenizer=loaded_tokenizer,
            do_login_first_time=script_args.do_login_first_time,
            huggingface_token=script_args.huggingface_token,
            huggingface_repo_name=script_args.huggingface_dataset_repo_name,
        )
    else:
        print("Skipping the push of the tokenizer to the hub...")

    if script_args.do_create_masked_encodings:

        print("Creating masked encodings of the train set...")
        train_set = create_masked_encodings(
            loaded_tokenizer, train_paths_list, TrainTextDataset, script_args.max_length
        )

        print("Saving the masked encodings of the train set...")
        save_masked_encodings(train_set, script_args.train_ecodings_file_path)

        print("Creating masked encodings of the test set...")
        test_dataset = create_masked_encodings(
            loaded_tokenizer, test_paths_list, TestTextDataset, script_args.max_length
        )

        print("Saving the masked encodings of the test set...")
        save_masked_encodings(test_dataset, script_args.test_encodings_file_path)

    # Train model
    if script_args.do_train_model:
        print("Loading the masked encodings of the train and test sets...")
        train_set = load_masked_encodings(script_args.train_ecodings_file_path)
        test_dataset = load_masked_encodings(script_args.test_encodings_file_path)

        if script_args.model_type == "bert":
            print("Training a BERT model using the PyTorch API...")

            ModelWrapper(
                train_set=train_set,
                test_set=test_dataset,
                model_type=script_args.model_type,
                vocab_size=script_args.vocab_size,
                max_length=script_args.max_length,
            )

        else:  # roberta
            print("Training a RoBeRTa model using the HuggingFace API...")

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=loaded_tokenizer,
                mlm=True,
                mlm_probability=0.15,
                # pad_to_multiple_of=8
            )

            ModelWrapper(
                train_set=train_set,
                test_set=test_dataset,
                data_collator=data_collator,
                model_type=script_args.model_type,
                vocab_size=script_args.vocab_size,
                max_length=script_args.max_length,
            )

    else:
        print("Skipping the training of a model from scratch...")


if __name__ == "__main__":
    main()

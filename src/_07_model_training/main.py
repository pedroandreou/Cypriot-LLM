import os

import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    RobertaConfig,
    RobertaForMaskedLM,
    set_seed,
)

from src._06_data_masking.masked_dataset import MaskedDataset
from src._07_model_training.training_methods.huggingface_model_trainer import (
    HuggingFaceTrainer,
)
from src._07_model_training.training_methods.pytorch_model_trainer import (
    PyTorchModelTrainer,
)
from src.utils.common_utils import echo_with_color, get_new_subdirectory_path

curr_dir = os.path.dirname(os.path.abspath(__file__))


def load_model(model_type: str, model_version: int):
    model_dir_path = os.path.join(
        curr_dir,
        "trained_model_bundle",
        f"cy{model_type}",
        f"model_v{model_version}",
    )

    if model_type == "bert":
        config = BertConfig.from_pretrained(model_dir_path)
        model = BertForMaskedLM.from_pretrained(model_dir_path, config=config)
    else:
        config = RobertaConfig.from_pretrained(model_dir_path)
        model = RobertaForMaskedLM.from_pretrained(model_dir_path, config=config)

    return model


def main(
    model_type: str,
    masked_encodings_version: int,
    trainer_type: str,
    seed: int,
    vocab_size: int,
    block_size: int,
    hidden_size: int,
    num_attention_heads: int,
    num_hidden_layers: int,
    type_vocab_size: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    num_train_epochs: int,
):
    # Create a directory for the model
    model_dir_path_w_model_type = os.path.join(
        curr_dir, "trained_model_bundle", f"cy{model_type}"
    )
    model_dir_path_w_model_type_n_version = get_new_subdirectory_path(
        model_dir_path_w_model_type, "model"
    )

    echo_with_color("Setting seed...", color="bright_cyan")
    set_seed(seed)

    echo_with_color("Detecting device...", color="bright_cyan")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # As we are training from scratch, we initialize from a config
    echo_with_color("Initializing config...", color="bright_cyan")
    ConfigClass = BertConfig if model_type == "bert" else RobertaConfig
    config = ConfigClass(
        vocab_size=vocab_size,
        max_position_embeddings=block_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
    )

    echo_with_color(
        "Initializing model from initialized config...", color="bright_cyan"
    )
    ModelClass = BertForMaskedLM if model_type == "bert" else RobertaForMaskedLM
    model = ModelClass(config=config).to(device)

    echo_with_color(
        "Loading the masked encodings of the train and test sets...",
        color="bright_cyan",
    )
    masked_train_dataset, masked_test_dataset = MaskedDataset().load_masked_encodings(
        model_type, masked_encodings_version
    )

    echo_with_color(
        f"Training model from scratch using {trainer_type.capitalize()} as our trainer type...",
        color="bright_cyan",
    )
    if trainer_type == "pytorch":
        pytorch_trainer = PyTorchModelTrainer(
            train_set=masked_train_dataset,
            test_set=masked_test_dataset,
            device=device,
            model_type=model_type,
            model=model,
            model_path=model_dir_path_w_model_type_n_version,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
        )
        pytorch_trainer.train()

    else:
        # should load the data collator here
        huggingface_trainer = HuggingFaceTrainer(
            train_set=masked_train_dataset,
            test_set=masked_test_dataset,
            model=model,
            model_path=model_dir_path_w_model_type_n_version,
            # data_collator=self.data_collator,
        )
        huggingface_trainer.train()

    # Should allow pushing the model to the hub here


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for model setup and training."
    )

    # Model
    parser.add_argument(
        "--model_type", type=str, default="bert", help="Type of model to use"
    )
    parser.add_argument(
        "--trainer_type",
        type=str,
        default="pytorch",
        help="Type of trainer to use: pytorch or huggingface",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    # Masked Encodings
    parser.add_argument(
        "--masked_encodings_version",
        type=int,
        default=1,
        help="Version of encodings to use",
    )

    # Model Configurations
    parser.add_argument("--vocab_size", type=int, default=30522)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--type_vocab_size", type=int, default=1)

    # Training Configurations
    parser.add_argument(
        "--train_batch_size", default=32, type=int, help="Training batch size."
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Evaluation batch size."
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="Learning rate for training."
    )
    parser.add_argument(
        "--num_train_epochs", default=3, type=int, help="Number of training epochs."
    )

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    args = parse_arguments()

    main(
        model_type=args.model_type,
        masked_encodings_version=args.masked_encodings_version,
        trainer_type=args.trainer_type,
        seed=args.seed,
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        type_vocab_size=args.type_vocab_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
    )

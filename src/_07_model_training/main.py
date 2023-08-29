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

"""
Following Intro_to_Weights_&_Biases Google Colab notebook
at https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb#scrollTo=xShwrFZeXPsL
"""


curr_dir = os.path.dirname(os.path.abspath(__file__))


def main(
    model_type: str,
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
    num_eval_epochs: int,
    # do_login_first_time: bool,
    # huggingface_token: str,
    # huggingface_repo_name: str,
):
    set_seed(seed)

    print("Loading the masked encodings of the train and test sets...")
    masked_train_set, masked_test_dataset = MaskedDataset().load_masked_encodings()

    print("Detect device...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Initialize model...")
    # Dynamically choose the correct configuration and model based on script_args.model_type
    ConfigClass = BertConfig if model_type == "bert" else RobertaConfig
    ModelClass = BertForMaskedLM if model_type == "bert" else RobertaForMaskedLM

    # As we are training from scratch, we initialize from a config
    print("Initialize config...")
    config = ConfigClass(
        vocab_size=vocab_size,
        max_position_embeddings=block_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
    )
    model = ModelClass(config=config).to(device)

    print(
        f"Training model from scratch using {trainer_type.capitalize()} as our trainer type"
    )
    model_path = os.path.join(curr_dir, "trained_model_bundle", f"cy{model_type}")
    if trainer_type == "pytorch":
        pytorch_trainer = PyTorchModelTrainer(
            train_set=masked_train_set,
            test_set=masked_test_dataset,
            device=device,
            model=model,
            model_path=model_path,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            num_eval_epochs=num_eval_epochs,
        )
        pytorch_trainer.train()

    else:
        # should load the data collator here
        huggingface_trainer = HuggingFaceTrainer(
            train_set=masked_train_set,
            test_set=masked_test_dataset,
            model=model,
            model_path=model_path,
            # data_collator=self.data_collator,
        )
        huggingface_trainer.train()

    print(model.num_parameters())

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
    parser.add_argument(
        "--num_eval_epochs", default=10, type=int, help="Number of evaluation epochs."
    )

    # parser.add_argument(
    #     "--do_login_first_time",
    #     action="store_true",
    #     help="Toggle first-time login. Credentials will be cached after the initial login to the hub.",
    # )
    # parser.add_argument(
    #     "--huggingface_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN")
    # )
    # parser.add_argument(
    #     "--huggingface_repo_name", type=str, default=os.getenv("HUGGINGFACE_REPO_NAME")
    # )

    return parser.parse_args()


if __name__ == "__main__":
    import argparse

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())

    args = parse_arguments()

    main(
        model_type=args.model_type,
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
        num_eval_epochs=args.num_eval_epochs,
        # do_login_first_time=args.do_login_first_time,
        # huggingface_token=args.huggingface_token,
        # huggingface_repo_name=args.huggingface_repo_name,
    )

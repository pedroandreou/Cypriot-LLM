import os
from dataclasses import dataclass, field
from typing import Optional
import torch

from transformers import AutoConfig, AutoModelForMaskedLM
from dotenv import find_dotenv, load_dotenv
from transformers import (
    HfArgumentParser,
    set_seed,
)
from src._05_data_tokenizing_and_masking.masked_dataset import MaskedDataset

from training_methods.huggingface_model_trainer import HuggingFaceTrainer
from training_methods.pytorch_model_trainer import PyTorchModelTrainer

load_dotenv(find_dotenv())


@dataclass
class ScriptArguments:
    model_type: str = field(default="bert", metadata={"help": "Type of model to use"})

    do_train_model: bool = field(default=False, metadata={"help": "Train model"})
    trainer_type: str = field(
        default="pytorch",
        metadata={"help": "Type of trainer to use: pytorch or huggingface"},
    )
    seed: int = field(default=42, metadata={"help": "Seed for reproducibility"})

    vocab_size: int = field(default=30522)
    block_size: str = field(default=512)
    hidden_size: int = field(default=768)
    num_attention_heads: int = field(default=12)
    num_hidden_layers: int = field(default=6)
    type_vocab_size: int = field(default=1)

    learning_rate: float = field(
        default=0.01, metadata={"help": "Learning Rate for the training"}
    )
    max_steps: int = field(
        default=1_000_000, metadata={"help": "The Number of Training steps to perform"}
    )

    do_login_first_time: bool = field(
        default=False,
        metadata={
            "help": "Toggle first-time login. Credentials will be cached after the initial login to the hub."
        },
    )
    huggingface_token: Optional[str] = field(default=os.getenv("HUGGINGFACE_TOKEN"))
    huggingface_repo_name: Optional[str] = field(
        default=os.getenv("HUGGINGFACE_REPO_NAME")
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # set seed for reproducibility
    set_seed(script_args.seed)

    # Train model
    if script_args.do_train_model:
        print("Loading the masked encodings of the train and test sets...")
        masked_train_set, masked_test_dataset = MaskedDataset().load_masked_encodings()

    #     model_path = f"trained_model_bundle/cy{script_args.model_type}"
    #     device = (
    #         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     )

    #     # As we are training from scratch, we initialize from a config
    #     # not from an existing pretrained model or checkpoint
    #     config = AutoConfig(
    #         vocab_size=script_args.vocab_size,  # tutorial example was 7015
    #         max_position_embeddings=script_args.block_size,  # tutorial example was 514
    #         hidden_size=script_args.hidden_size,
    #         num_attention_heads=script_args.num_attention_heads,
    #         num_hidden_layers=script_args.num_hidden_layers,
    #         type_vocab_size=script_args.type_vocab_size,
    #     )
    #     model = AutoModelForMaskedLM(config=config).to(device)

    #     # Print the model parameters
    #     print(model.num_parameters())

    #     print(f"Training model from scratch using the {script_args.trainer_type}.capitalize() as the trainer type")
    #     if script_args.trainer_type == "pytorch":
    #         PyTorchModelTrainer(
    #             device=device,
    #             model=model,
    #             model_path=model_path,
    #             train_set=masked_train_set,
    #             test_set=masked_test_dataset,
    #         )

    #     else: # huggingface
    #         # should load the data collator here

    #         HuggingFaceTrainer(
    #             device=device,
    #             model=model,
    #             model_path=model_path,
    #             train_set=masked_train_set,
    #             test_set=masked_test_dataset,
    #             # data_collator=self.data_collator,
    #         )

    # else:
    #     print("Skipping the training of a model from scratch...")


if __name__ == "__main__":
    main()

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from dotenv import find_dotenv, load_dotenv
from training_methods.huggingface_model_trainer import HuggingFaceTrainer
from training_methods.pytorch_model_trainer import PyTorchModelTrainer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    HfArgumentParser,
    RobertaConfig,
    RobertaForMaskedLM,
    set_seed,
)

from _06_data_masking.masked_dataset import MaskedDataset

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


curr_dir = os.path.dirname(os.path.abspath(__file__))


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

        print("Detect device...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print("Initialize model...")
        # Dynamically choose the correct configuration and model based on script_args.model_type
        ConfigClass = BertConfig if script_args.model_type == "bert" else RobertaConfig
        ModelClass = (
            BertForMaskedLM if script_args.model_type == "bert" else RobertaForMaskedLM
        )

        # As we are training from scratch, we initialize from a config
        """
        Following Intro_to_Weights_&_Biases Google Colab notebook
        at https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb#scrollTo=xShwrFZeXPsL
        """
        config = ConfigClass(
            vocab_size=script_args.vocab_size,
            max_position_embeddings=script_args.block_size,
            hidden_size=script_args.hidden_size,
            num_attention_heads=script_args.num_attention_heads,
            num_hidden_layers=script_args.num_hidden_layers,
            type_vocab_size=script_args.type_vocab_size,
        )
        model = ModelClass(config=config).to(device)

        print(
            f"Training model from scratch using {script_args.trainer_type}.capitalize() as our trainer type"
        )
        model_path = os.path.join(
            curr_dir, "trained_model_bundle", f"cy{script_args.model_type}"
        )
        if script_args.trainer_type == "pytorch":
            pytorch_trainer = PyTorchModelTrainer(
                train_set=masked_train_set,
                test_set=masked_test_dataset,
                device=device,
                model=model,
                model_path=model_path,
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

    else:
        print("Skipping the training of a model from scratch...")
        # Load saved model here

    # Should allow pushing the model to the hub here


if __name__ == "__main__":
    main()

import os

import torch
from dotenv import find_dotenv, load_dotenv
from transformers import AutoConfig, AutoModelForMaskedLM

from .training_methods.huggingface_model_trainer import HuggingFaceTrainer
from .training_methods.pytorch_model_trainer import PyTorchModelTrainer

load_dotenv(find_dotenv())


class ModelWrapper:
    def __init__(
        self,
        train_set,
        test_set,
        data_collator=None,
        model_path: str = os.getenv("MODEL_DIR_PATH"),
        model_type="bert",
        vocab_size=30_522,
        max_length=512,
    ):
        self.train_set = train_set
        self.test_set = test_set
        self.data_collator = data_collator  # Optional as this is way that RoBeRTa is trained using the HuggingFace API works

        self.model_path = model_path
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # As we are training from scratch, we initialize from a config
        # not from an existing pretrained model or checkpoint
        config = AutoConfig(
            vocab_size=vocab_size,  # tutorial example was 7015
            max_position_embeddings=max_length,  # tutorial example was 514
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        self.model = AutoModelForMaskedLM(config=config).to(self.device)

        # Print the model parameters
        print(self.model.num_parameters())

        # Train model
        if model_type == "bert":
            PyTorchModelTrainer(
                model=self.model,
                train_set=self.train_set,
                device=self.device,
                model_path=self.model_path,
            )

        else:  # roberta
            HuggingFaceTrainer(
                model=self.model,
                train_set=self.train_set,
                test_set=self.test_set,
                data_collator=self.data_collator,
                model_path=self.model_path,
            )

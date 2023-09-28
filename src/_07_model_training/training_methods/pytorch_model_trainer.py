import os
from typing import Tuple, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import wandb
from dotenv import find_dotenv, load_dotenv
from torch.optim import AdamW
from tqdm import tqdm

load_dotenv(find_dotenv())

WANDB_KEY = os.getenv("WANDB_KEY")

curr_dir = os.path.dirname(os.path.realpath(__file__))


class PyTorchModelTrainer:
    def __init__(
        self,
        train_set,
        test_set,
        device,
        model_type,
        model,
        model_path,
        tokenizer_type,
        do_apply_logit_norm,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        num_train_epochs,
    ):
        self.train_set = train_set
        self.test_set = test_set

        self.device = device
        self.model_type = model_type
        self.model = model
        self.model_path = model_path

        self.tokenizer_type = tokenizer_type

        self.do_apply_logit_norm = do_apply_logit_norm

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.learning_rate = learning_rate

        self.num_train_epochs = num_train_epochs

    def logit_norm_pytorch(self, input_tensor, axis=-1):
        DEFAULT_EPSILON = 1e-4
        x = input_tensor
        x_denominator = x.square()
        x_denominator_sum = x_denominator.sum(dim=axis, keepdim=True)
        x_denominator = (
            torch.sqrt(x_denominator_sum + DEFAULT_EPSILON) + DEFAULT_EPSILON
        )

        return x / x_denominator

    def train(self):
        """
        PyTorch is considered as the manual way to train since it is used in combination with the manual implementation of the MLM task
        This is because the automatic implementation of the MLM task uses a data collator which is only used with the HuggingFace API
        """

        wandb.login(key=WANDB_KEY)

        # Initialize Weights and Biases
        config = {
            "model": self.model_type,
            "model_path": self.model_path,
            "tokenizer": self.tokenizer_type,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_train_epochs": self.num_train_epochs,
            "total_trainable_params": self.model.num_parameters(),
        }
        wandb.init(project=f"cypriot-model", config=config)

        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )

        optim = AdamW(
            self.model.parameters(), lr=self.learning_rate
        )  # Initialize optimizer

        accumulation_steps = 8

        for epoch in tqdm(
            range(self.num_train_epochs), leave=True, total=self.num_train_epochs
        ):
            losses = []

            # setup loop with TQDM and dataloader
            pbar = tqdm(train_loader, leave=True)
            for i, batch in enumerate(pbar):
                # Activate training mode
                self.model.train()

                # pull all tensor batches required for training
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if not self.do_apply_logit_norm:
                    # Forward pass
                    outputs = self.model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs.loss  # extract loss

                else:
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    normalized_logits = self.logit_norm_pytorch(outputs.logits)
                    criterion = torch.nn.CrossEntropyLoss()  # compute loss
                    loss = criterion(
                        normalized_logits.view(-1, normalized_logits.size(-1)),
                        labels.view(-1),
                    )

                # Backward pass
                loss.backward()

                if not self.do_apply_logit_norm:
                    # Perform weight update
                    optim.step()

                    # Zero out gradients
                    optim.zero_grad()

                else:  # Apply gradient accumulation
                    # Check if it's time for a weight update
                    if (i + 1) % accumulation_steps == 0:
                        # Divide accumulated graadients by accumulation steps
                        for param in self.model.parameters():
                            param.grad /= accumulation_steps

                        # Perform weight update
                        optim.step()

                        # Zero out gradients
                        for param in self.model.parameters():
                            param.grad.zero_()

                # Log the loss at each step
                wandb.log({"batch_loss": loss.item()})

                # Print relevant info to progress bar
                pbar.set_description(f"Epoch {epoch}")
                pbar.set_postfix(loss=loss.item())
                losses.append(loss.item())

            # Log the mean loss at the end of each epoch
            mean_train_loss = np.mean(losses)
            wandb.log({"mean_train_loss": mean_train_loss})

            # Evaluate the model on the test set after each training epoch
            mean_test_loss = self.eval()
            wandb.log({"mean_test_loss": mean_test_loss})

        # Finish Weights and Biases
        wandb.finish()

        # Save model
        self.model.save_pretrained(self.model_path)

    def eval(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

        self.model.eval()  # set model to evaluation mode
        losses = []

        pbar = tqdm(test_loader, leave=True)

        # No gradient computation during evaluation
        with torch.no_grad():
            for batch in pbar:
                # copy input to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if not self.do_apply_logit_norm:
                    # Forward pass
                    outputs = self.model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs.loss  # extract loss

                else:
                    # Forward pass
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    normalized_logits = self.logit_norm_pytorch(outputs.logits)
                    criterion = torch.nn.CrossEntropyLoss()  # compute the loss
                    loss = criterion(
                        normalized_logits.view(-1, normalized_logits.size(-1)),
                        labels.view(-1),
                    )

                # output current loss
                pbar.set_postfix(loss=loss.item())
                losses.append(loss.item())

        mean_test_loss = np.mean(losses)
        return mean_test_loss

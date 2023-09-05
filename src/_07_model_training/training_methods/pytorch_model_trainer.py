import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW

from src.utils.common_utils import echo_with_color
import wandb
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

WANDB_KEY = os.getenv("WANDB_KEY")


class PyTorchModelTrainer:
    def __init__(
        self,
        train_set,
        test_set,
        device,
        model,
        model_path,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        num_train_epochs,
    ):
        self.train_set = train_set
        self.test_set = test_set

        self.device = device
        self.model = model
        self.model_path = model_path

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.learning_rate = learning_rate

        self.num_train_epochs = num_train_epochs

    def train(self):
        """
        PyTorch is considered as the manual way to train since it is used in combination with the manual implementation of the MLM task
        This is because the automatic implementation of the MLM task uses a data collator which is only used with the HuggingFace API
        """

        wandb.login(key=WANDB_KEY)

        # Initialize Weights and Biases
        config = {
            "model": self.model.__class__.__name__,
            "model_path": self.model_path,
            "learning_rate": self.learning_rate,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_train_epochs": self.num_train_epochs,
        }
        wandb.init(project="nlp-model-training", config=config)

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

        for epoch in range(self.num_train_epochs):
            losses = []

            # setup loop with TQDM and dataloader
            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                self.model.train()  # Activate training mode

                # initialize calculated gradients (from prev step)
                optim.zero_grad()

                # pull all tensor batches required for training
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # process
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )

                # extract loss
                loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()

                # Log the batch loss to Weights and Biases after each step
                wandb.log({"batch_loss": loss.item()})

                # print relevant info to progress bar
                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())
                losses.append(loss.item())

            # Log the mean loss at the end of each epoch
            mean_train_loss = np.mean(losses)
            wandb.log({"epoch_loss": mean_train_loss})
            print("Mean Training Loss", mean_train_loss)

            # Evaluate the model on the test set after each training epoch
            mean_test_loss = self.eval()
            wandb.log({"mean_test_loss": mean_test_loss})

        # Finish Weights and Biases
        wandb.finish()

        # Save model
        self.model.save_pretrained(self.model_path)
        echo_with_color(
            f"The model are parameters are:\n {self.model.num_parameters()}",
            color="bright_cyan",
        )

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

        loop = tqdm(test_loader, leave=True)

        # iterate over dataset
        for batch in loop:
            # copy input to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # predict
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels
            )

            # extract loss
            loss = outputs.loss

            # output current loss
            loop.set_postfix(loss=loss.item())
            losses.append(loss.item())

        mean_test_loss = np.mean(losses)
        print("Mean Test Loss", mean_test_loss)
        return mean_test_loss

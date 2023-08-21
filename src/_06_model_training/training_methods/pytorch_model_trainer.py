import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW


class PyTorchModelTrainer:
    def __init__(self, train_set, test_set, device, model, model_path):
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=16, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=16, shuffle=True
        )

        self.device = device
        self.model = model
        self.model_path = model_path

    def train(self):
        """
        PyTorch is considered as the manual way to train since it is used in combination with the manual implementation of the MLM task
        This is because the automatic implementation of the MLM task uses a data collator which is only used with the HuggingFace API below
        """
        self.model.train()  # Activate training mode
        optim = AdamW(self.model.parameters(), lr=1e-4)  # Initialize optimizer

        epochs = 2
        for epoch in range(epochs):
            losses = []

            # setup loop with TQDM and dataloader
            loop = tqdm(self.train_loader, leave=True)
            for batch in loop:
                print(batch)
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
                # print relevant info to progress bar
                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())
                losses.append(loss.item())

            print("Mean Training Loss", np.mean(losses))

        # Save model
        self.model.save_pretrained(self.model_path)

    def eval(self):
        # set model to evaluation mode
        self.model.eval()
        losses = []

        epochs = 10
        for epoch in range(epochs):
            loop = tqdm(self.test_loader, leave=True)

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
                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())
                losses.append(loss.item())
            print("Mean Test Loss", np.mean(losses))

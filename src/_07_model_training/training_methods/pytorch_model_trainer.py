import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW


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
        num_eval_epochs,
    ):
        self.train_set = train_set
        self.train_set = test_set

        self.device = device
        self.model = model
        self.model_path = model_path

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.num_eval_epochs = num_eval_epochs

    def train(self):
        """
        PyTorch is considered as the manual way to train since it is used in combination with the manual implementation of the MLM task
        This is because the automatic implementation of the MLM task uses a data collator which is only used with the HuggingFace API
        """

        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )

        self.model.train()  # Activate training mode
        optim = AdamW(
            self.model.parameters(), lr=self.learning_rate
        )  # Initialize optimizer

        for epoch in range(self.num_train_epochs):
            losses = []

            # setup loop with TQDM and dataloader
            loop = tqdm(train_loader, leave=True)
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
        test_loader = torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )

        self.model.eval()  # set model to evaluation mode
        losses = []

        for epoch in range(self.num_eval_epochs):
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
                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())
                losses.append(loss.item())
            print("Mean Test Loss", np.mean(losses))

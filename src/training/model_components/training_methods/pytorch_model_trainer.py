import torch
from tqdm import tqdm
from transformers import AdamW


class PyTorchModelTrainer:
    def __init__(self, model, train_set, device, model_path):
        self.model = model
        self.train_set = train_set
        self.device = device
        self.model_path = model_path

        # Train model
        self.train()

    def train(self):
        """
        PyTorch is considered as the manual way to train since it is used in combination with the manual implementation of the MLM task
        This is because the automatic implementation of the MLM task uses a data collator which is only used with the HuggingFace API below
        """
        self.model.train()  # Activate training mode
        optim = AdamW(self.model.parameters(), lr=1e-4)  # Initialize optimizer
        loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=16, shuffle=True
        )  # Build a DataLoader using PyTorch

        epochs = 2
        for epoch in range(epochs):
            # setup loop with TQDM and dataloader
            loop = tqdm(loader, leave=True)
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

        # Save model
        self.model.save_pretrained(self.model_path)

    def train_and_evaluate(self):
        epochs = 10

        for epoch in range(epochs):
            loop = tqdm(loader, leave=True)

            # set model to training mode
            self.model.train()
            losses = []

            # iterate over dataset
            for batch in loop:
                self.optim.zero_grad()

                # copy input to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # predict
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )

                # update weights
                loss = outputs.loss
                loss.backward()

                self.optim.step()

                # output current loss
                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())
                losses.append(loss.item())

            print("Mean Training Loss", np.mean(losses))
            losses = []
            loop = tqdm(test_loader, leave=True)

            # set model to evaluation mode
            self.model.eval()

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

                # update weights
                loss = outputs.loss

                # output current loss
                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())
                losses.append(loss.item())
            print("Mean Test Loss", np.mean(losses))

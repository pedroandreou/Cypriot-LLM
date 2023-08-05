import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    RobertaConfig,
    RobertaForMaskedLM,
    Trainer,
    TrainingArguments,
)


class ModelWrapper:
    def __init__(
        self,
        train_set,
        test_set,
        data_collator=None,
        model_path=None,
        model_type="bert",
        vocab_size=30_522,
        max_length=512,
    ):
        self.train_set = train_set
        self.test_set = test_set

        self.data_collator = data_collator  # Optional as this is way that RoBeRTa is trained using the HuggingFace API works
        self.model_path = model_path  # Optional as this is way that RoBeRTa is trained using the HuggingFace API works

        self.model_type = model_type
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # As we are training from scratch, we initialize from a config
        # not from an existing pretrained model or checkpoint
        if self.model_type == "bert":
            config = BertConfig(
                vocab_size=vocab_size, max_position_embeddings=max_length
            )
            self.model = BertForMaskedLM(config=config).to(self.device)

        else:  # roberta
            config = RobertaConfig(
                vocab_size=vocab_size,  # tutorial example was 7015
                max_position_embeddings=max_length,  # tutorial example was 514
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=6,
                type_vocab_size=1,
            )
            self.model = RobertaForMaskedLM(config=config).to(self.device)

        # Print the model parameters
        print(self.model.num_parameters())

        # Train model
        if self.model_type == "bert":
            self.train_model_using_pytorch_api()

        else:  # roberta
            self.train_model_using_huggingface_trainer_api()

    def train_model_using_pytorch_api(self):
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

        # Return model
        return self.model

    def train_model_using_huggingface_trainer_api(self):
        """
        This training method is considered as automatic since it is used in combination with the automatic implementation of the MLM task
        by using a data collator as one of its training arguments
        """

        # Since we have set logging_steps and save_steps to 1000,
        # then the trainer will evaluate and save the model after every 1000 steps
        # (i.e trained on steps x gradient_accumulation_step x per_device_train_size = 1000x8x10 = 80,000 samples).
        # As a result, I have canceled the training after about 19 hours of training,
        # or 10000 steps (that is about 1.27 epochs, or trained on 800,000 samples), and started to use the model.

        # Initialize training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,  # output directory to where save model checkpoint
            evaluation_strategy="steps",  # evaluate each `logging_steps` steps
            overwrite_output_dir=True,
            num_train_epochs=10,  # number of training epochs, feel free to tweak
            per_device_train_batch_size=10,  # the training batch size, put it as high as your GPU memory fits
            gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
            per_device_eval_batch_size=64,  # evaluation batch size
            logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
            save_steps=1000,
            # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
            # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
        )

        # Initialize the trainer and pass everything to it
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=self.train_set,
            eval_dataset=self.test_set,
        )

        # Train the model
        trainer.train()

        # Return model
        return self.model

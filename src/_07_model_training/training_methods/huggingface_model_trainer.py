from transformers import Trainer, TrainingArguments


class HuggingFaceTrainer:
    def __init__(self, train_set, test_set, model, model_path, data_collator=None):
        self.train_set = train_set
        self.test_set = test_set
        self.data_collator = data_collator
        self.model = model
        self.model_path = model_path

    def train(self):
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
            output_dir=self.model_path,  # script_args.repository_id, - # output directory to where save model checkpoint
            overwrite_output_dir=True,
            per_device_train_batch_size=10,  # the training batch size, put it as high as your GPU memory fits
            per_device_eval_batch_size=64,  # evaluation batch size
            num_train_epochs=10,  # number of training epochs, feel free to tweak
            # learning_rate=script_args.learning_rate,
            # seed=seed,
            # max_steps=script_args.max_steps,
            gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
            ### logging & evaluation strategies ###
            # logging_dir=f"{script_args.repository_id}/logs",
            # logging_strategy="steps",
            logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
            # save_strategy="steps",
            evaluation_strategy="steps",  # evaluate each `logging_steps` steps
            save_steps=1000,  # 5_000
            save_total_limit=3,  # 2 - # whether you don't have much space so you let only 3 model weights saved in the disk
            # report_to="tensorboard",
            ### push to hub parameters ###
            # push_to_hub=True,
            # hub_strategy="every_save",
            # hub_model_id=script_args.repository_id,
            load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
            ### pretraining ###
            # ddp_find_unused_parameters=True,
            # throughput_warmup_steps=2,
        )

        # Initialize the trainer and pass everything to it
        trainer = Trainer(
            model=self.model,
            args=training_args,
            # data_collator=self.data_collator, # do I need to pass the data collator too?
            train_dataset=self.train_set,
            # eval_dataset=self.test_set,
            # tokenizer=tokenizer, # do I need to pass the tokenizer too?
        )

        # Train the model
        trainer.train()

        # Save model
        self.model.save_pretrained(self.model_path)

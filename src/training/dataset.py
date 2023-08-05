import torch
from mlm_task import MLMTask


class BaseDataset(MLMTask):
    def __init__(
        self,
        tokenizer,
        file_paths,
        model_type="bert",
        max_length=512,
    ):
        MLMTask.__init__(self, tokenizer, model_type)

        self.tokenizer = tokenizer
        self.max_len = max_length

        self.texts = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                self.texts.append(file.read())
        self.dataset = self._create_masked_dataset()

    def _create_masked_dataset(self):
        encodings = self.tokenizer(
            self.texts,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True,
        )
        encodings_dict = {
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
            "labels": torch.tensor(encodings["input_ids"]).detach().clone(),
            "special_tokens_mask": torch.tensor(encodings["special_tokens_mask"]),
        }
        # Apply MLM masking
        masked_encodings = self.process_batch(encodings_dict)
        return masked_encodings

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, index):
        """Get item from the Dataset"""
        item = {key: val[index] for key, val in self.dataset.items()}
        return item

    def get_data_collator(self):
        """Return data collator from the MLM task as its the superclass of this class"""
        return self.data_collator


class TrainTextDataset(BaseDataset):
    def __init__(self, tokenizer, file_paths, model_type="bert", max_length=512):
        super().__init__(tokenizer, file_paths, model_type, max_length)


class TestTextDataset(BaseDataset):
    def __init__(self, tokenizer, file_paths, model_type="bert", max_length=512):
        super().__init__(tokenizer, file_paths, model_type, max_length)

import os

import torch
from joblib import load
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling


class MaskedDataset(Dataset):
    def __init__(
        self,
        tokenized_dataset=None,
        model_type: str = None,
        mlm_type: str = None,
        mlm_probability: float = None,
    ):
        self.masked_encodings = None

        if (
            tokenized_dataset is None
            and model_type is None
            and mlm_type is None
            and mlm_probability is None
        ):
            self.default_constructor()
        else:
            self.parameterized_constructor(
                tokenized_dataset, model_type, mlm_type, mlm_probability
            )

    def default_constructor(self):
        print(
            "Using default constructor. This instance is meant for loading data only."
        )
        pass

    def parameterized_constructor(
        self, tokenized_dataset, model_type, mlm_type, mlm_probability
    ):
        self.encodings = tokenized_dataset
        self.model_type = model_type
        self.mlm_type = mlm_type
        self.mlm_probability = mlm_probability
        self.masked_encodings = self._create_masked_dataset()

    def _create_masked_dataset(self):
        masked_encodings_list = {"input_ids": [], "attention_mask": [], "labels": []}

        for i in range(len(self.encodings)):  # Loop over each tensor in the dataset
            input_ids = self.encodings[i]
            attention_mask = (input_ids != 0).long()
            labels = input_ids.detach().clone()

            encodings_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            masked_encodings = self.process_batch(encodings_dict)

            for key in masked_encodings:
                masked_encodings_list[key].append(masked_encodings[key])

        # Convert lists to tensors
        for key in masked_encodings_list:
            masked_encodings_list[key] = torch.stack(masked_encodings_list[key])

        return masked_encodings_list

    def process_batch(self, data):
        if self.mlm_type == "manual":
            return self.manual_mlm(data)
        else:
            return self.automatic_mlm(data)

    def manual_mlm(self, data):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        # create random array of floats with equal dims to input_ids
        rand = torch.rand(input_ids.shape)

        # mask random 15% where token is not [CLS], [PAD], [SEP]
        if self.model_type == "bert":
            mask_arr = (
                (rand < self.mlm_probability)
                & (input_ids != 0)
                & (input_ids != 1)
                & (input_ids != 2)
            )
        else:
            mask_arr = (
                (rand < self.mlm_probability)
                & (input_ids != 100)
                & (input_ids != 101)
                & (input_ids != 102)  # not sure for the 100
            )

        selection = torch.flatten(mask_arr.nonzero()).tolist()

        if self.model_type == "bert":
            input_ids[selection] = 3
        else:
            input_ids[selection] = 103

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def automatic_mlm(self, data):
        pass
        # data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=loaded_tokenizer,
        #     mlm=True,
        #     mlm_probability=self.mlm_probability,
        #     # pad_to_multiple_of=8
        # )

        # return self.data_collator(data)

    def __len__(self):
        if self.masked_encodings is None:
            print("Warning: Dataset not initialized. Returning length 0.")
            return 0
        return len(self.masked_encodings["input_ids"])

    def __getitem__(self, index):
        if self.masked_encodings is None:
            print("Warning: Dataset not initialized. Returning empty item.")
            return {}
        item = {key: val[index] for key, val in self.masked_encodings.items()}
        return item

    def load_masked_encodings(self):
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        train_dataset_path = os.path.join(
            curr_dir, "saved_data", "masked_encodings", "masked_train_dataset.pkl"
        )
        test_dataset_path = os.path.join(
            curr_dir, "saved_data", "masked_encodings", "masked_test_dataset.pkl"
        )

        train_dataset = load(train_dataset_path)
        test_dataset = load(test_dataset_path)

        return train_dataset, test_dataset

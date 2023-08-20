import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling


class MaskedDataset(Dataset):
    def __init__(
        self,
        tokenized_dataset,
        model_type: str = "bert",
        mlm_type: str = "manual",
        mlm_probability: float = 0.15,
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
        return len(self.masked_encodings["input_ids"])

    def __getitem__(self, index):
        """Get item from the Dataset"""
        item = {key: val[index] for key, val in self.masked_encodings.items()}
        return item

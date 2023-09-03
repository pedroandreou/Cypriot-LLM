import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
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

        if all(
            arg is None
            for arg in (tokenized_dataset, model_type, mlm_type, mlm_probability)
        ):
            self.default_constructor()
        else:
            self.parameterized_constructor(
                tokenized_dataset, model_type, mlm_type, mlm_probability
            )

    def default_constructor(self):
        print(
            "Using default constructor. This instance of MaskedDataset class is meant for loading data only."
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
        total_masked_encodings_dict = {"input_ids": [], "attention_mask": [], "labels": []}
        
        count = 0

        for i in tqdm(
            range(len(self.encodings)), desc="Masking encodings"
        ):  # Loop over each tensor in the dataset
            
            labels = self.encodings[i].clone()
            mask = (labels != 0).long()
            input_ids = labels.detach().clone()

            encodings_dict = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "labels": labels,
            }

            
            if self.mlm_type == "manual":
                masked_encodings_dict = self.manual_mlm(encodings_dict)

            else:  # automatic
                masked_encodings_dict = self.automatic_mlm(encodings_dict)
        

            for key in masked_encodings_dict:
                total_masked_encodings_dict[key].append(masked_encodings_dict[key])

        # Convert lists to tensors
        for key in total_masked_encodings_dict[key]:
            total_masked_encodings_dict[key] = torch.stack(total_masked_encodings_dict[key])

        return total_masked_encodings_dict


    def manual_mlm(self, batch):
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["labels"]


        # create random array of floats with equal dims to input_ids
        rand = torch.rand(input_ids.shape)

        # mask random 15% where token is not [CLS], [PAD], [SEP]
        if self.model_type == "bert":
            mask_arr = (
                (rand < self.mlm_probability)
                * (input_ids != 101)
                * (input_ids != 102)
                * (input_ids != 0)
            )
        else:
            mask_arr = (
                (rand < self.mlm_probability)
                * (input_ids != 1)
                * (input_ids != 2)
                * (input_ids != 0)
            )

        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr.nonzero()).tolist()

            # mask input ids
            if self.model_type == "bert":
                input_ids[selection] = 103
            else:
                input_ids[selection] = 3

        return {
            "input_ids": input_ids,
            "attention_mask": mask,
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
        return self.masked_encodings["input_ids"].shape[0]

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.masked_encodings.items()}

    @staticmethod
    def load_masked_encodings():
        def get_dataset_path(set_type):
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            folder_name = "masked_encodings"
            filename = f"masked_{set_type}_dataset.pth"

            return os.path.join(curr_dir, folder_name, filename)

        train_dataset_path = get_dataset_path("train")
        train_dataset = torch.load(train_dataset_path)

        test_dataset_path = get_dataset_path("test")
        test_dataset = torch.load(test_dataset_path)

        return train_dataset, test_dataset

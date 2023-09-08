import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# from transformers import DataCollatorForLanguageModeling

curr_dir = os.path.dirname(os.path.realpath(__file__))


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
        self._create_masked_dataset()

    def _create_masked_dataset(self):
        self.masked_encodings = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for i in tqdm(
            range(len(self.encodings)), desc="Masking encodings"
        ):  # Loop over each tensor in the dataset

            current_tensor_input_ids = self.encodings[i]["input_ids"].clone()
            current_tensor_mask = self.encodings[i]["attention_mask"].clone()
            current_tensor_labels = self.encodings[i]["input_ids"].clone()

            if self.mlm_type == "manual":
                current_tensor_masked_input_ids = self.manual_mlm(
                    current_tensor_input_ids
                )
            else:  # automatic
                current_tensor_masked_input_ids = self.automatic_mlm(
                    current_tensor_input_ids
                )

            current_tensor_encodings = {
                "input_ids": current_tensor_masked_input_ids,
                "attention_mask": current_tensor_mask,
                "labels": current_tensor_labels,
            }

            for key in current_tensor_encodings:
                self.masked_encodings[key].append(current_tensor_encodings[key])

        for key in self.masked_encodings:
            self.masked_encodings[key] = torch.stack(self.masked_encodings[key])

    def manual_mlm(self, input_ids):
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

        for _ in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr.nonzero()).tolist()

            # mask input ids
            if self.model_type == "bert":
                input_ids[selection] = 103
            else:
                input_ids[selection] = 3

        # masked input ids
        return input_ids

    def automatic_mlm(self, data):
        # data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=loaded_tokenizer,
        #     mlm=True,
        #     mlm_probability=self.mlm_probability,
        #     # pad_to_multiple_of=8
        # )

        # return self.data_collator(data)
        pass

    def __repr__(self):
        tokenizer_type = (
            "BertWordPieceTokenizer"
            if self.model_type == "bert"
            else "ByteLevelBPETokenizer"
        )
        return f"<TokenizedDataset: ModelType={self.model_type}, TokenizerType={tokenizer_type}, NumExamples={len(self.masked_encodings['input_ids'])}>"

    def __len__(self):
        return self.masked_encodings["input_ids"].shape[0]

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.masked_encodings.items()}

    ##############################
    ### Load Encodings Methods ###
    ##############################
    def _get_dataset_path(
        self, model_type: str, set_type: str, masked_encodings_version: int
    ):
        folder_name = os.path.join(
            "masked_encodings",
            f"cy{model_type}",
            f"masked_encodings_v{masked_encodings_version}",
        )
        filename = f"masked_{set_type}_dataset.pth"

        return os.path.join(curr_dir, folder_name, filename)

    def load_and_set_train_encodings(
        self, model_type: str, masked_encodings_version: int
    ):
        self.model_type = model_type
        train_dataset_path = self._get_dataset_path(
            self.model_type, "train", masked_encodings_version
        )
        self.masked_encodings = torch.load(train_dataset_path)

        return self.masked_encodings

    def load_and_set_test_encodings(
        self, model_type: str, masked_encodings_version: int
    ):
        self.model_type = model_type
        test_dataset_path = self._get_dataset_path(
            self.model_type, "test", masked_encodings_version
        )
        self.masked_encodings = torch.load(test_dataset_path)

        return self.masked_encodings

    def load_masked_encodings(self, model_type: str, masked_encodings_version: int):
        train_set = self.load_and_set_train_encodings(
            model_type, masked_encodings_version
        )
        test_set = self.load_and_set_test_encodings(
            model_type, masked_encodings_version
        )

        return train_set, test_set

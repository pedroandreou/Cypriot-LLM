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
        tokenizer_type: str = None,
        mlm_type: str = None,
        mlm_probability: float = None,
    ):
        self.masked_encodings = None

        if all(
            arg is None
            for arg in (tokenized_dataset, tokenizer_type, mlm_type, mlm_probability)
        ):
            self.default_constructor()
        else:
            self.parameterized_constructor(
                tokenized_dataset, tokenizer_type, mlm_type, mlm_probability
            )

    def default_constructor(self):
        print(
            "Using default constructor. This instance of MaskedDataset class is meant for loading data only."
        )
        pass

    def parameterized_constructor(
        self, tokenized_dataset, tokenizer_type, mlm_type, mlm_probability
    ):
        self.encodings = tokenized_dataset
        self.tokenizer_type = tokenizer_type
        self.mlm_type = mlm_type
        self.mlm_probability = mlm_probability
        self._create_masked_dataset()

    def _create_masked_dataset(self):
        labels = torch.stack([x["input_ids"] for x in self.encodings])
        mask = torch.stack([x["attention_mask"] for x in self.encodings])
        input_ids = labels.detach().clone()

        if self.mlm_type == "static":
            masked_input_ids = self.manual_static_masking(input_ids)

            self.masked_encodings = {
                "input_ids": masked_input_ids,
                "attention_mask": mask,
                "labels": labels,
            }

        else:  # dynamic
            pass

    def manual_static_masking(self, input_ids):
        rand = torch.rand(input_ids.shape)

        # mask random 15% where token is not [CLS], [PAD], [SEP]
        if self.tokenizer_type == "WP":
            mask_arr = (
                (rand < self.mlm_probability)
                * (input_ids != 2)
                * (input_ids != 0)
                * (input_ids != 3)
            )
        else:  # BPE
            mask_arr = (
                (rand < self.mlm_probability)
                * (input_ids != 1)
                * (input_ids != 0)
                * (input_ids != 2)
            )

        # loop through each row in input_ids tensor (cannot do in parallel)
        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()

            # mask input_ids
            input_ids[i, selection] = 4  # same for both WP and BPE

        return input_ids

    def automatic_dynamic_masking(self, data):
        # data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=loaded_tokenizer,
        #     mlm=True,
        #     mlm_probability=self.mlm_probability,
        #     # pad_to_multiple_of=8
        # )

        # return self.data_collator(data)
        pass

    def __repr__(self):
        real_tokenizer_type = (
            "BertWordPieceTokenizer"
            if self.tokenizer_type == "WP"
            else "ByteLevelBPETokenizer"
        )
        return f"<MaskedDataset: TokenizerType={real_tokenizer_type}, NumExamples={len(self.masked_encodings['input_ids'])}>"

    def __len__(self):
        return self.masked_encodings["input_ids"].shape[0]

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.masked_encodings.items()}

    ##############################
    ### Load Encodings Methods ###
    ##############################
    def _get_dataset_path(
        self, tokenizer_type: str, set_type: str, masked_encodings_version: int
    ):
        folder_name = os.path.join(
            "masked_encodings",
            f"cy{tokenizer_type}",
            f"masked_encodings_v{masked_encodings_version}",
        )
        filename = f"masked_{set_type}_dataset.pth"

        return os.path.join(curr_dir, folder_name, filename)

    def load_and_set_train_encodings(
        self, tokenizer_type: str, masked_encodings_version: int
    ):
        self.tokenizer_type = tokenizer_type
        train_set_path = self._get_dataset_path(
            self.tokenizer_type, "train", masked_encodings_version
        )
        self.masked_encodings = torch.load(train_set_path)

        return self.masked_encodings

    def load_and_set_test_encodings(
        self, tokenizer_type: str, masked_encodings_version: int
    ):
        self.tokenizer_type = tokenizer_type
        test_set_path = self._get_dataset_path(
            self.tokenizer_type, "test", masked_encodings_version
        )
        self.masked_encodings = torch.load(test_set_path)

        return self.masked_encodings

    def load_masked_encodings(self, tokenizer_type: str, masked_encodings_version: int):
        train_set = self.load_and_set_train_encodings(
            tokenizer_type, masked_encodings_version
        )
        test_set = self.load_and_set_test_encodings(
            tokenizer_type, masked_encodings_version
        )

        return train_set, test_set

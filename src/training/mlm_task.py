import torch
from transformers import DataCollatorForLanguageModeling


class MLMTask:
    def __init__(
        self,
        tokenizer,
        model_type="bert",
    ):
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.data_collator = None

    def bert_mlm(self, batch):
        labels = batch["input_ids"].clone().detach()
        mask = batch["input_ids"].clone().detach()

        # make a copy of labels tensor, this will be input_ids
        input_ids = labels.clone().detach()

        # create random array of floats with equal dims to input_ids
        rand = torch.rand(input_ids.shape)

        # mask random 15% where token is not 0 [CLS], 1 [PAD], or 2 [SEP]
        mask_arr = (
            (rand < 0.15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
        )

        # loop through each row in input_ids tensor (cannot do in parallel)
        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            # mask input_ids
            input_ids[i, selection] = 4  # our custom [MASK] token == 4

        return {
            "input_ids": input_ids,
            "attention_mask": mask,
            "labels": labels,
            "special_tokens_mask": batch["special_tokens_mask"],
        }

    def roberta_mlm(self, batch):
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        return self.data_collator(batch)

    def process_batch(self, batch):
        if self.model_type == "bert":
            return self.bert_mlm(batch)
        else:  # roberta
            return self.roberta_mlm(batch)

from typing import Optional, Tuple, Union

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


class TextQADataset:
    def __init__(
        self,
        data_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        max_token_length: int = 1024,
        ignore_index: int = -100,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ) -> None:
        data_dir = dataset_name if data_dir is None else data_dir
        if dataset_name == "pubmed_qa":
            self.dataset = load_dataset(data_dir, "pqa_labeled", split="train")
        else:
            raise NotImplementedError

        self.length = len(self.dataset)
        self.max_token_length = max_token_length
        self.ignore_index = ignore_index
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.dataset[int(idx)]
        input_ids, labels, attention_mask = preprocess(
            record, self.tokenizer, max_length=self.max_token_length, ignore_index=self.ignore_index
        )
        return input_ids, labels, attention_mask


def preprocess(
    example, tokenizer: PreTrainedTokenizer, max_length: int = 512, ignore_index: int = -100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    question = example["question"]
    answer = example["long_answer"]

    conversation = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # set generation prompt to be false, following suggestion from https://huggingface.co/docs/transformers/main/en/chat_templating#model-training
    text = tokenizer.apply_chat_template(conversation=conversation, tokenize=False, add_generation_prompt=False)

    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
        add_special_tokens=True,
    )

    # prepare inputs and the corresponding padding mask
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0].astype(np.bool_)

    # prepare labels, only the text in the answer content is a valid label
    labels = input_ids.copy()

    content_begin_token = tokenizer.vocab.get("<|im_start|>")
    # right shift 3 tokens as the answer start
    answer_begin_idx = np.where(input_ids == content_begin_token)[0][2].item() + 3
    labels[:answer_begin_idx] = ignore_index

    padding_token = tokenizer.vocab.get(tokenizer.special_tokens_map["pad_token"])
    # left shift 1 tokens as the answer end
    padding_begin_idx = np.where(input_ids == padding_token)[0][0].item() - 1
    labels[padding_begin_idx:] = ignore_index

    return input_ids, labels, attention_mask

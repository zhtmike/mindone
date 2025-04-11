import argparse
import logging
import os
import sys
from typing import Tuple

from transformers import AutoTokenizer

import mindspore as ms
import mindspore.nn as nn
from mindspore.communication import get_group_size, get_rank, init

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.transformers.models.qwen2 import Qwen2ForCausalLM
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5 Inference script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", default=42, type=int, help="Training seed.")
    parser.add_argument("--mode", default=1, choices=[0, 1], help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).")
    parser.add_argument("--jit_level", default="O1", choices=["O0", "O1"], help="Jit Level")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name.")
    parser.add_argument("--output_path", default="./output", help="Output directory to save the inference result.")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel training.")
    args = parser.parse_args()
    return args


def init_env(args) -> Tuple[int, int]:
    ms.set_seed(args.seed)
    ms.set_context(mode=args.mode, jit_config=dict(jit_level=args.jit_level))
    if args.use_parallel:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
    else:
        device_num, rank_id = 1, 0

    return device_num, rank_id


def main(args):
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # init env
    _, rank_id = init_env(args)
    set_logger(output_dir=os.path.join(args.output_path, "logs"), rank=rank_id)

    # prepare model
    logger.info("Creating model `%s`", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    with nn.no_init_parameters():
        model = Qwen2ForCausalLM.from_pretrained(args.model_name)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="np").input_ids
    input_ids = ms.Tensor(input_ids, ms.int32)

    input_kwargs = {}
    input_kwargs["input_ids"] = input_ids

    output_ids = model.generate(**input_kwargs, use_cache=False, max_new_tokens=512, do_sample=False)
    output_ids = output_ids.asnumpy()

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    args = parse_args()
    main(args)

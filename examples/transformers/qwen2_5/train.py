import argparse
import logging
import os
import sys
import time
from typing import Tuple

from transformers import Qwen2Config

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.amp import auto_mixed_precision
from mindspore.communication import get_group_size, get_rank, init
from mindspore.dataset import GeneratorDataset

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)


from dataset import TextQADataset

from mindone.trainers import create_optimizer
from mindone.trainers.checkpoint import CheckpointManager
from mindone.transformers import Qwen2ForCausalLM
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5 Training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", default=42, type=int, help="Training seed.")
    parser.add_argument("--mode", default=1, choices=[0, 1], help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).")
    parser.add_argument("--jit_level", default="O1", choices=["O0", "O1"], help="Jit Level")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel training.")
    parser.add_argument("--output_path", default="./output", help="Output directory to save the training result.")
    parser.add_argument("--dataset_name", default="pubmed_qa", help="Dataset Name.")
    parser.add_argument("--max_token_length", default=1024, type=int, help="Maximum token length.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name.")
    parser.add_argument(
        "--load_weight", default=True, type=str2bool, help="Load pretrained weight or random initialization."
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp32", "bf16"],
        help="If it is not fp32, then train with auto mixed precision.",
    )
    parser.add_argument(
        "--optim", default="adamw_mint", type=str, choices=["adamw_mint", "came"], help="Optimizer name."
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Warmup steps.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of total training epochs.")
    parser.add_argument("--batch_size", default=4, type=int, help="Training batch size.")
    parser.add_argument("--ckpt_max_keep", default=3, type=int, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--clip_grad", default=True, type=str2bool, help="Clip gradient.")
    parser.add_argument("--clip_grad_value", default=1.0, type=float, help="Clip gradient value.")
    args = parser.parse_args()
    return args


def init_env(args) -> Tuple[int, int]:
    ms.set_seed(args.seed)
    ms.set_context(mode=args.mode, jit_config=dict(jit_level=args.jit_level))
    if args.use_parallel:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num
        )
    else:
        device_num, rank_id = 1, 0

    return device_num, rank_id


def main(args):
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # init env
    device_num, rank_id = init_env(args)
    set_logger(output_dir=os.path.join(args.output_path, "logs"), rank=rank_id)

    # prepare model
    logger.info("Creating model `%s`", args.model_name)
    config = Qwen2Config.from_pretrained(args.model_name)
    if args.load_weight:
        with nn.no_init_parameters():
            model = Qwen2ForCausalLM.from_pretrained(args.model_name, config=config)
    else:
        logger.info("Initialize network randomly.")
        model = Qwen2ForCausalLM(config)
    model.set_train()

    if args.dtype != "fp32":
        logger.info("Using AMP with data type %s", args.dtype)
        dtype = ms.bfloat16 if args.dtype == "bf16" else ms.float16
        model = auto_mixed_precision(model, amp_level="auto", dtype=dtype)

    # prepare dataset
    logger.info("Creating dataset `%s`", args.dataset_name)
    dataset = TextQADataset(
        dataset_name=args.dataset_name, max_token_length=args.max_token_length, tokenizer_name=args.model_name
    )
    data_generator = GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        column_types=[ms.int64, ms.int64, ms.bool_],
        shuffle=True,
        num_parallel_workers=4,
        num_shards=device_num,
        shard_id=rank_id,
    )
    data_generator = data_generator.batch(args.batch_size, drop_remainder=True, num_parallel_workers=2)

    # prepare trainer
    lr = nn.WarmUpLR(learning_rate=args.learning_rate, warmup_steps=args.warmup_steps)
    optimizer = create_optimizer(model.trainable_params(), args.optim, lr=lr, group_strategy="not_grouping")

    train_step_func = ms.value_and_grad(model, None, optimizer.parameters)
    reduce_func = nn.DistributedGradReducer(optimizer.parameters) if args.use_parallel else lambda x: x

    ds_iter = data_generator.create_dict_iterator(num_epochs=-1)
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)

    logger.info("Start training...")
    for epoch in range(1, args.epochs + 1):
        for step, data in enumerate(ds_iter):
            start_time_s = time.time()
            loss, grads = train_step_func(**data)
            grads = reduce_func(grads)
            if args.clip_grad:
                grads = ops.clip_by_global_norm(grads, clip_norm=args.clip_grad_value)
            optimizer(grads)
            step_time = time.time() - start_time_s
            logger.info(
                "epoch %d, step %d, loss %.8f, lr %.7f, step time %.2fms",
                epoch,
                step,
                loss.item(),
                optimizer.learning_rate(optimizer.global_step - 1)[0].item(),
                step_time * 1000,
            )
        ckpt_name = f"model-epoch-{epoch}.ckpt"
        ckpt_manager.save(model, None, ckpt_name=ckpt_name, append_dict=None)


if __name__ == "__main__":
    args = parse_args()
    main(args)

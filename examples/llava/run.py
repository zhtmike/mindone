#!/usr/bin/env python
import copy
import json
import logging
import os
import time
from typing import Any, Dict

from model.llava_next import LlavaNextForConditionalGeneration
from PIL import Image
from pipeline import TextGenerator
from transformers import LlavaNextProcessor

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


def load_network(config: Dict[str, Any], ckpt_path: str, use_cache: bool = False) -> nn.Cell:
    config_ = copy.copy(config)
    config_["vision_config"]["hidden_size"] = 1024
    config_["text_config"]["hidden_size"] = 4096

    # for debuging
    # config_["vision_config"]["num_hidden_layers"] = 1
    # config_["text_config"]["num_hidden_layers"] = 1

    vision_config = config_.pop("vision_config")
    text_config = config_.pop("text_config")
    logging.info("Init Model...")
    network = LlavaNextForConditionalGeneration(
        vision_config,
        text_config,
        past_key_value_cache=use_cache,
        attn_implementation="eager",
        dtype=ms.float16,
        **config_,
    )
    logging.info("Loading the checkpoint...")
    ms.load_checkpoint(ckpt_path, net=network, strict_load=True)
    return network


def main():
    MODEL_PATH = "/mnt/disk4/mikecheung/data/model/llava-v1.6-mistral-7b-hf-slim"
    with open(os.path.join(MODEL_PATH, "config.json"), "r") as f:
        config = json.load(f)

    # prepare image and text prompt, using the appropriate prompt template
    processor = LlavaNextProcessor.from_pretrained(MODEL_PATH)
    image = Image.open("llava_v1_5_radar.jpg")
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

    inputs = processor(prompt, image, return_tensors="np")
    inputs = {k: Tensor(v) for k, v in inputs.items()}

    # autoregressively complete prompt
    print("=" * 60)
    print("KV Cache:")
    network = load_network(config, "llava_1_6.ckpt", use_cache=True)
    pipeline = TextGenerator(network, max_new_tokens=100)
    start = time.time()
    output = pipeline.generate(**inputs)
    end = time.time()
    print(processor.decode(output[0], skip_special_tokens=True))
    print(f"Time Taken: {end-start:.3f}")
    print("=" * 60)

    print("Non KV Cache:")
    network = load_network(config, "llava_1_6.ckpt", use_cache=False)
    pipeline = TextGenerator(network, max_new_tokens=100)
    start = time.time()
    output = pipeline.generate(**inputs)
    end = time.time()
    print(processor.decode(output[0], skip_special_tokens=True))
    print(f"Time Taken: {end-start:.3f}")


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    main()

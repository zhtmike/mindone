# Adapted from https://github.com/bytedance/MVDream/blob/main/mvdream/model_zoo.py

""" Utiliy functions to load pre-trained models more easily """
import os

import pkg_resources
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

import mindspore as ms

from .ldm.util import instantiate_from_config

PRETRAINED_MODELS = {
    "sd-v2.1-base-4view": {
        "config": "sd-v2-base.yaml",
        "repo_id": "MVDream/MVDream",
        "filename": "sd-v2.1-base-4view.pt",
    },
    "sd-v1.5-4view": {"config": "sd-v1.yaml", "repo_id": "MVDream/MVDream", "filename": "sd-v1.5-4view.pt"},
}


def get_config_file(config_path):
    cfg_file = pkg_resources.resource_filename("mvdream", os.path.join("configs", config_path))
    if not os.path.exists(cfg_file):
        raise RuntimeError(f"Config {config_path} not available!")
    return cfg_file


def build_model(model_name, ckpt_path=None, cache_dir=None):
    if model_name not in PRETRAINED_MODELS:
        raise RuntimeError(
            f"Model name {model_name} is not a pre-trained model. Available models are:\n- "
            + "\n- ".join(PRETRAINED_MODELS.keys())
        )
    model_info = PRETRAINED_MODELS[model_name]

    # Instiantiate the model
    print(f"Loading model from config: {model_info['config']}")
    config_file = get_config_file(model_info["config"])
    config = OmegaConf.load(config_file)
    model = instantiate_from_config(config.model)

    # Load pre-trained checkpoint from huggingface
    if not ckpt_path:
        ckpt_path = hf_hub_download(repo_id=model_info["repo_id"], filename=model_info["filename"], cache_dir=cache_dir)
        print(f"Loading model from cache file: {ckpt_path}")
    m, u = ms.load_param_into_net(model, ms.load_checkpoint(ckpt_path), strict_load=False)
    # assert m == [], f"there is not loaded param {m}"
    # assert u == [], f"there is not loaded ckpt weight {u}"
    # print(f"there is not loaded ckpt weight {u}")
    return model

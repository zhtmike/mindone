# Adapted from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py
# common functions for training

import argparse
import ast
import bisect
import glob
import hashlib
import importlib
import json
import logging
import math
import os
import pathlib
import random
import re
import shutil
import subprocess
import time
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import imagesize
import library.model_util as model_util
import library.sai_model_spec as sai_model_spec
import numpy as np
import toml
from library import custom_train_functions
from library.utils import setup_logging

# from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

import mindspore as ms
from mindspore import nn, ops
from mindspore.communication import get_group_size, get_local_rank, get_rank, init
from mindspore.dataset import transforms, vision

from mindone.diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from mindone.diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType
from mindone.transformers import CLIPTextModel, CLIPTextModelWithProjection

setup_logging()

logger = logging.getLogger(__name__)


def set_seed(seed=42, rank=0):
    ms.set_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def is_master(args):
    return args.rank == 0


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.distributed:
        init()
        args.local_rank = get_local_rank()
        args.world_size = get_group_size()
        args.rank = get_rank()
        ms.context.set_auto_parallel_context(
            device_num=args.world_size,
            global_rank=args.rank,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )

    device = f"{ms.get_context('device_target')}:{ms.get_context('device_id')}"
    args.device = device
    return device


def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False


# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"  # ここからtokenizerだけ使う v2とv2.1はtokenizer仕様は同じ

HIGH_VRAM = False

# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

# # region dataset

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

try:
    import pillow_avif  # noqa: F401

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:  # noqa: E722
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin  # noqa: F401

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:  # noqa: E722
    pass

# JPEG-XL on Windows
try:
    import pillow_jxl  # noqa: F401

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:  # noqa: E722
    pass

IMAGE_TRANSFORMS = transforms.Compose(
    [
        vision.ToTensor(),
        vision.Normalize([0.5], [0.5], is_hwc=False),
    ]
)

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"


class ImageInfo:
    def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: ms.Tensor = None
        self.latents_flipped: ms.Tensor = None
        self.latents_npz: str = None
        self.latents_original_size: Tuple[int, int] = None  # original image size, not latents size
        self.latents_crop_ltrb: Tuple[
            int, int
        ] = None  # crop left top right bottom in original pixel size, not latents size
        self.cond_img_path: str = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        # SDXL, optional
        self.text_encoder_outputs_npz: Optional[str] = None
        self.text_encoder_outputs1: Optional[ms.Tensor] = None
        self.text_encoder_outputs2: Optional[ms.Tensor] = None
        self.text_encoder_pool2: Optional[ms.Tensor] = None


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "the max_size should be larger than the min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # 前処理時は (image_key, image, original size, crop left/top)、学習時は image_key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # logger.info(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 拡大および縮小を行う
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # logger.info(f"use predef, {image_width}, {image_height}, {reso}, {resized_size}")
        else:
            # 縮小のみを行う
            if image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # logger.info(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # logger.info(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # logger.info(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # logger.info(f"use arbitrary {image_width}, {image_height}, {resized_size}, {bucket_width}, {bucket_height}")

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucketのほうが横長→縦を合わせる
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    # albumentationsへの依存をなくしたがとりあえず同じinterfaceを持たせる
    # 消除对 albumentations 的依赖，但暂时保留相同的界面

    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                # hue shift
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                # random gamma
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image**gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(self, use_color_aug: bool):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseSubset:
    def __init__(
        self,
        image_dir: Optional[str],
        num_repeats: int,
        shuffle_caption: bool,
        caption_separator: str,
        keep_tokens: int,
        keep_tokens_separator: str,
        secondary_separator: Optional[str],
        enable_wildcard: bool,
        color_aug: bool,
        flip_aug: bool,
        face_crop_aug_range: Optional[Tuple[float, float]],
        random_crop: bool,
        caption_dropout_rate: float,
        caption_dropout_every_n_epochs: int,
        caption_tag_dropout_rate: float,
        caption_prefix: Optional[str],
        caption_suffix: Optional[str],
        token_warmup_min: int,
        token_warmup_step: Union[float, int],
    ) -> None:
        self.image_dir = image_dir
        self.num_repeats = num_repeats
        self.shuffle_caption = shuffle_caption
        self.caption_separator = caption_separator
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator
        self.secondary_separator = secondary_separator
        self.enable_wildcard = enable_wildcard
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix

        self.token_warmup_min = token_warmup_min  # step=0におけるタグの数
        self.token_warmup_step = token_warmup_step  # N（N<1ならN*max_train_steps）ステップ目でタグの数が最大になる

        self.img_count = 0


class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        cache_info: bool,
        num_repeats,
        shuffle_caption,
        caption_separator: str,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir


class FineTuningSubset(BaseSubset):
    def __init__(
        self,
        image_dir,
        metadata_file: str,
        num_repeats,
        shuffle_caption,
        caption_separator,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        assert metadata_file is not None, "metadata_file must be specified / metadata_fileは指定が必須です"

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
        )

        self.metadata_file = metadata_file

    def __eq__(self, other) -> bool:
        if not isinstance(other, FineTuningSubset):
            return NotImplemented
        return self.metadata_file == other.metadata_file


class ControlNetSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        conditioning_data_dir: str,
        caption_extension: str,
        cache_info: bool,
        num_repeats,
        shuffle_caption,
        caption_separator,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
        )

        self.conditioning_data_dir = conditioning_data_dir
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, ControlNetSubset):
            return NotImplemented
        return self.image_dir == other.image_dir and self.conditioning_data_dir == other.conditioning_data_dir


# class BaseDataset(torch.utils.data.Dataset):
class BaseDataset:
    def __init__(
        self,
        tokenizer: Union[CLIPTokenizer, List[CLIPTokenizer]],
        max_token_length: int,
        resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        debug_dataset: bool,
    ) -> None:
        super().__init__()

        self.tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        self.max_token_length = max_token_length
        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[Union[DreamBoothSubset, FineTuningSubset]] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.tokenizer_max_length = (
            self.tokenizers[0].model_max_length if max_token_length is None else max_token_length + 2
        )

        self.current_epoch: int = 0  # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, Union[DreamBoothSubset, FineTuningSubset]] = {}

        self.replacements = {}

        # caching
        self.caching_mode = None  # None, 'latents', 'text'

    def set_seed(self, seed):
        self.seed = seed

    def set_caching_mode(self, mode):
        self.caching_mode = mode

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:  # epochが切り替わったらバケツをシャッフルする
            self.shuffle_buckets()
        self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, subset: BaseSubset, caption):
        # caption に prefix/suffix を付ける
        if subset.caption_prefix:
            caption = subset.caption_prefix + " " + caption
        if subset.caption_suffix:
            caption = caption + " " + subset.caption_suffix

        # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
        is_drop_out = subset.caption_dropout_rate > 0 and random.random() < subset.caption_dropout_rate
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            # process wildcards
            if subset.enable_wildcard:
                # if caption is multiline, random choice one line
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))

                # wildcard is like '{aaa|bbb|ccc...}'
                # escape the curly braces like {{ or }}
                replacer1 = "⦅"
                replacer2 = "⦆"
                while replacer1 in caption or replacer2 in caption:
                    replacer1 += "⦅"
                    replacer2 += "⦆"

                caption = caption.replace("{{", replacer1).replace("}}", replacer2)

                # replace the wildcard
                def replace_wildcard(match):
                    return random.choice(match.group(1).split("|"))

                caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

                # unescape the curly braces
                caption = caption.replace(replacer1, "{").replace(replacer2, "}")
            else:
                # if caption is multiline, use the first line
                caption = caption.split("\n")[0]

            if subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
                fixed_tokens = []
                flex_tokens = []
                fixed_suffix_tokens = []
                if (
                    hasattr(subset, "keep_tokens_separator")
                    and subset.keep_tokens_separator
                    and subset.keep_tokens_separator in caption
                ):
                    fixed_part, flex_part = caption.split(subset.keep_tokens_separator, 1)
                    if subset.keep_tokens_separator in flex_part:
                        flex_part, fixed_suffix_part = flex_part.split(subset.keep_tokens_separator, 1)
                        fixed_suffix_tokens = [
                            t.strip() for t in fixed_suffix_part.split(subset.caption_separator) if t.strip()
                        ]

                    fixed_tokens = [t.strip() for t in fixed_part.split(subset.caption_separator) if t.strip()]
                    flex_tokens = [t.strip() for t in flex_part.split(subset.caption_separator) if t.strip()]
                else:
                    tokens = [t.strip() for t in caption.strip().split(subset.caption_separator)]
                    flex_tokens = tokens[:]
                    if subset.keep_tokens > 0:
                        fixed_tokens = flex_tokens[: subset.keep_tokens]
                        flex_tokens = tokens[subset.keep_tokens :]

                if subset.token_warmup_step < 1:  # 初回に上書きする
                    subset.token_warmup_step = math.floor(subset.token_warmup_step * self.max_train_steps)
                if subset.token_warmup_step and self.current_step < subset.token_warmup_step:
                    tokens_len = (
                        math.floor(
                            (self.current_step)
                            * ((len(flex_tokens) - subset.token_warmup_min) / (subset.token_warmup_step))
                        )
                        + subset.token_warmup_min
                    )
                    flex_tokens = flex_tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    l = []  # noqa: E741
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            l.append(token)
                    return l

                if subset.shuffle_caption:
                    random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

            # process secondary separator
            if subset.secondary_separator:
                caption = caption.replace(subset.secondary_separator, subset.caption_separator)

            # textual inversion対応
            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if type(str_to) == list:
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        input_ids = tokenizer(
            caption, padding="max_length", truncation=True, max_length=self.tokenizer_max_length, return_tensors="np"
        ).input_ids
        input_ids = ms.Tensor(input_ids)

        if self.tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(
                    1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = ops.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(
                    1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2
                ):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = ops.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = ops.stack(iids_list)  # 3,77
        return input_ids

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def make_buckets(self):
        """
        bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
        min_size and max_size are ignored when enable_bucket is False
        """
        logger.info("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        if self.enable_bucket:
            logger.info("make buckets")
        else:
            logger.info("prepare dataset")

        # bucketを作成し、画像をbucketに振り分ける
        if self.enable_bucket:
            if self.bucket_manager is None:  # fine tuningの場合でmetadataに定義がある場合は、すでに初期化済み
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                )
                if not self.bucket_no_upscale:
                    self.bucket_manager.make_buckets()
                else:
                    logger.warning(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, \
                            because bucket reso is defined by image size automatically / \
                            bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
                    )

            img_ar_errors = []
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(
                    image_width, image_height
                )

                # logger.info(image_info.image_key, image_info.bucket_reso)
                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(False, (self.width, self.height), None, None, None)
            self.bucket_manager.set_predefined_resos([(self.width, self.height)])  # ひとつの固定サイズbucketのみ
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(
                    image_width, image_height
                )

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(image_info.bucket_reso, image_info.image_key)

        # bucket情報を表示、格納する
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            logger.info("number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）")
            for i, (reso, bucket) in enumerate(zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {"resolution": reso, "count": len(bucket)}
                    logger.info(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            img_ar_errors = np.array(img_ar_errors)
            mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            logger.info(f"mean ar error (without repeats): {mean_img_ar_error}")

        # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
        self.buckets_indices: List(BucketBatchIndex) = []  # type: ignore
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(BucketBatchIndex(bucket_index, self.batch_size, batch_index))

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def verify_bucket_reso_steps(self, min_steps: int):
        assert self.bucket_reso_steps is None or self.bucket_reso_steps % min_steps == 0, (
            f"bucket_reso_steps is {self.bucket_reso_steps}. it must be divisible by {min_steps}.\n"
            + f"bucket_reso_stepsが{self.bucket_reso_steps}です。{min_steps}で割り切れる必要があります"
        )

    def is_latent_cacheable(self):
        return all([not subset.color_aug and not subset.random_crop for subset in self.subsets])

    def is_text_encoder_output_cacheable(self):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0
                    or subset.shuffle_caption
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    # TODO
    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        raise NotImplementedError

    # weight_dtypeを指定するとText Encoderそのもの、およひ出力がweight_dtypeになる
    # SDXLでのみ有効だが、datasetのメソッドとする必要があるので、sdxl_train_util.pyではなくこちらに実装する
    # SD1/2に対応するにはv2のフラグを持つ必要があるので後回し
    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        raise NotImplementedError

    def get_image_size(self, image_path):
        return imagesize.get(image_path)

    def load_image_with_face_info(self, subset: BaseSubset, image_path: str):
        img = load_image(image_path)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # いい感じに切り出す
    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # 画像サイズはsizeより大きいのでリサイズする
        face_size = max(face_w, face_h)
        size = min(self.height, self.width)  # 短いほう
        min_scale = max(self.height / height, self.width / width)  # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
        min_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1])))  # 指定した顔最小サイズ
        max_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0])))  # 指定した顔最大サイズ
        if min_scale >= max_scale:  # range指定がmin==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # 顔を中心として448*640とかへ切り出す
        for axis, (target_size, length, face_p) in enumerate(
            zip((self.height, self.width), (height, width), (face_cy, face_cx))
        ):
            p1 = face_p - target_size // 2  # 顔を中心に持ってくるための切り出し位置

            if subset.random_crop:
                # 背景も含めるために顔を中心に置く確率を高めつつずらす
                range = max(length - face_p, face_p)  # 画像の端から顔中心までの距離の長いほう
                p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range  # -range ~ +range までのいい感じの乱数
            else:
                # range指定があるときのみ、すこしだけランダムに（わりと適当）
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        if self.caching_mode is not None:  # return batch for latents/text encoder outputs caching
            return self.get_item_for_caching(bucket, bucket_batch_size, image_index)

        loss_weights = []
        captions = []
        input_ids_list = []
        input_ids2_list = []
        latents_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs1_list = []
        text_encoder_outputs2_list = []
        text_encoder_pool2_list = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]
            loss_weights.append(
                self.prior_loss_weight if image_info.is_reg else 1.0
            )  # in case of fine tuning, is_reg is always False

            flipped = subset.flip_aug and random.random() < 0.5  # not flipped or flipped with 50% chance

            # image/latentsを処理する
            if image_info.latents is not None:  # cache_latents=Trueの場合
                original_size = image_info.latents_original_size
                crop_ltrb = image_info.latents_crop_ltrb  # calc values later if flipped
                if not flipped:
                    latents = image_info.latents
                else:
                    latents = image_info.latents_flipped

                image = None
            # TODO - cache latent
            elif image_info.latents_npz is not None:  # FineTuningDatasetまたはcache_latents_to_disk=Trueの場合
                raise NotImplementedError

            else:
                # 画像を読み込み、必要ならcropする
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(subset, image_info.absolute_path)
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img, original_size, crop_ltrb = trim_and_resize_if_required(
                        subset.random_crop, img, image_info.bucket_reso, image_info.resized_size
                    )
                else:
                    if face_cx > 0:  # 顔位置情報あり
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / \
                            画像サイズが大きいのでface_crop_aug_rangeかrandom_crop、またはbucketを有効にしてください: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]
                        # note: we add the resize operation for small image here
                        # instead of directly raise "image size is small" error as kohya org repo does.
                    elif im_h == self.height or im_w != self.width:
                        img = cv2.resize(img, (self.height, self.width))
                        logger.info(
                            f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}, \
                                    {(im_h, im_w)} resized to {(self.height, self.width)}"
                        )

                    # notes: we find the org repo have bugs in getting the `original_size` but we just follow it.
                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    img = aug(image=img)["image"]

                if flipped:
                    img = img[:, ::-1, :].copy()  # copy to avoid negative stride problem

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0のtorch.Tensorになる

            image = ms.Tensor(image) if image is not None else image
            images.append(image)
            latents_list.append(latents)

            target_size = (
                (image.shape[2], image.shape[1]) if image is not None else (latents.shape[2] * 8, latents.shape[1] * 8)
            )

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # captionとtext encoder outputを処理する
            caption = image_info.caption  # default
            if image_info.text_encoder_outputs1 is not None:
                text_encoder_outputs1_list.append(image_info.text_encoder_outputs1)
                text_encoder_outputs2_list.append(image_info.text_encoder_outputs2)
                text_encoder_pool2_list.append(image_info.text_encoder_pool2)
                captions.append(caption)
            elif image_info.text_encoder_outputs_npz is not None:
                text_encoder_outputs1, text_encoder_outputs2, text_encoder_pool2 = load_text_encoder_outputs_from_disk(
                    image_info.text_encoder_outputs_npz
                )
                text_encoder_outputs1_list.append(text_encoder_outputs1)
                text_encoder_outputs2_list.append(text_encoder_outputs2)
                text_encoder_pool2_list.append(text_encoder_pool2)
                captions.append(caption)
            else:
                caption = self.process_caption(subset, image_info.caption)
                if self.XTI_layers:
                    caption_layer = []
                    for layer in self.XTI_layers:
                        token_strings_from = " ".join(self.token_strings)
                        token_strings_to = " ".join([f"{x}_{layer}" for x in self.token_strings])
                        caption_ = caption.replace(token_strings_from, token_strings_to)
                        caption_layer.append(caption_)
                    captions.append(caption_layer)
                else:
                    captions.append(caption)

                if not self.token_padding_disabled:  # this option might be omitted in future
                    if self.XTI_layers:
                        token_caption = self.get_input_ids(caption_layer, self.tokenizers[0])
                    else:
                        token_caption = self.get_input_ids(caption, self.tokenizers[0])
                    input_ids_list.append(token_caption)

                    if len(self.tokenizers) > 1:
                        if self.XTI_layers:
                            token_caption2 = self.get_input_ids(caption_layer, self.tokenizers[1])
                        else:
                            token_caption2 = self.get_input_ids(caption, self.tokenizers[1])
                        input_ids2_list.append(token_caption2)

        example = {}
        example["loss_weights"] = ms.Tensor(loss_weights)

        # note: ms2.5 GeneratorDataset.create_dict_iterator raise error in None value scenario,
        # so we do not return key of None value here
        if len(text_encoder_outputs1_list) == 0:
            if self.token_padding_disabled:
                # padding=True means pad in the batch
                example["input_ids"] = self.tokenizer[0](
                    captions, padding=True, truncation=True, return_tensors="np"
                ).input_ids
                if len(self.tokenizers) > 1:
                    example["input_ids2"] = self.tokenizer[1](
                        captions, padding=True, truncation=True, return_tensors="np"
                    ).input_ids

            else:
                example["input_ids"] = ops.stack(input_ids_list)
                example["input_ids2"] = ops.stack(input_ids2_list) if len(self.tokenizers) > 1 else None

        else:
            example["text_encoder_outputs1_list"] = ops.stack(text_encoder_outputs1_list)
            example["text_encoder_outputs2_list"] = ops.stack(text_encoder_outputs2_list)
            example["text_encoder_pool2_list"] = ops.stack(text_encoder_pool2_list)

        if images[0] is not None:
            images = ops.stack(images)
            images = images.float()
            # images = images.to(memory_format=torch.contiguous_format).float()
            example["images"] = images

        if latents_list[0] is not None:
            example["latents"] = ops.stack(latents_list)
        example["captions"] = captions

        example["original_sizes_hw"] = ops.stack([ms.Tensor(x) for x in original_sizes_hw])
        example["crop_top_lefts"] = ops.stack([ms.Tensor(x) for x in crop_top_lefts])
        example["target_sizes_hw"] = ops.stack([ms.Tensor(x) for x in target_sizes_hw])
        example["flippeds"] = flippeds

        example["network_multipliers"] = ms.Tensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            if flip_aug is None:
                flip_aug = subset.flip_aug
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
                assert random_crop == subset.random_crop, "random_crop must be same in a batch"
                assert bucket_reso == image_info.bucket_reso, "bucket_reso must be same in a batch"

            caption = image_info.caption  # TODO cache some patterns of dropping, shuffling, etc.

            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset: bool,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, network_multiplier, debug_dataset)

        assert resolution is not None, "resolution is required / resolution（解像度）指定は必須です"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            assert (
                min(resolution) >= min_bucket_reso
            ), "min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。\
                解像度を大きくするかmin_bucket_resoを小さくしてください"
            assert (
                max(resolution) <= max_bucket_reso
            ), "max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。\
                解像度を小さくするかmin_bucket_resoを大きくしてください"
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension, enable_wildcard):
            # captionの候補ファイル名を作る
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            logger.error(f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {cap_path}")
                            raise e
                        assert len(lines) > 0, f"caption file is empty / キャプションファイルが空です: {cap_path}"
                        if enable_wildcard:
                            caption = "\n".join([line.strip() for line in lines if line.strip() != ""])  # 空行を除く、改行で連結
                        else:
                            caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"not directory: {subset.image_dir}")
                return [], []

            info_cache_file = os.path.join(subset.image_dir, self.IMAGE_INFO_CACHE_FILE)
            use_cached_info_for_subset = subset.cache_info
            if use_cached_info_for_subset:
                logger.info(f"using cached image info for this subset / このサブセットで、キャッシュされた画像情報を使います: {info_cache_file}")
                if not os.path.isfile(info_cache_file):
                    logger.warning(
                        "image info file not found. You can ignore this warning if this is the first time to use this subset"
                        + " / キャッシュファイルが見つかりませんでした。初回実行時はこの警告を無視してください: {metadata_file}"
                    )
                    use_cached_info_for_subset = False

            if use_cached_info_for_subset:
                # json: {`img_path`:{"caption": "caption...", "resolution": [width, height]}, ...}
                with open(info_cache_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                img_paths = list(metas.keys())
                sizes = [meta["resolution"] for meta in metas.values()]

                # we may need to check image size and existence of image files, but it takes time, so user should check it before training
            else:
                img_paths = glob_images(subset.image_dir, "*")
                sizes = [None] * len(img_paths)

            logger.info(f"found directory {subset.image_dir} contains {len(img_paths)} image files")

            if use_cached_info_for_subset:
                captions = [meta["caption"] for meta in metas.values()]
                missing_captions = [
                    img_path for img_path, caption in zip(img_paths, captions) if caption is None or caption == ""
                ]
            else:
                # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
                captions = []
                missing_captions = []
                for img_path in img_paths:
                    cap_for_img = read_caption(img_path, subset.caption_extension, subset.enable_wildcard)
                    if cap_for_img is None and subset.class_tokens is None:
                        logger.warning(
                            f"neither caption file nor class tokens are found. use empty caption for {img_path} / \
                                キャプションファイルもclass tokenも見つかりませんでした。空のキャプションを使用します: {img_path}"
                        )
                        captions.append("")
                        missing_captions.append(img_path)
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            missing_captions.append(img_path)
                        else:
                            captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)  # タグ頻度を記録

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = number_of_missing_captions - number_of_missing_captions_to_show

                logger.warning(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. \
                        If class token exists, it will be used. /"
                    + f"{number_of_missing_captions}枚の画像にキャプションファイルが見つかりませんでした。これらの画像についてはキャプションなしで学習を続行します。\
                        class tokenが存在する場合はそれを使います。"
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        logger.warning(missing_caption + f"... and {remaining_missing_captions} more")
                        break
                    logger.warning(missing_caption)

            if not use_cached_info_for_subset and subset.cache_info:
                logger.info(f"cache image info for / 画像情報をキャッシュします : {info_cache_file}")
                sizes = [self.get_image_size(img_path) for img_path in tqdm(img_paths, desc="get image size")]
                matas = {}
                for img_path, caption, size in zip(img_paths, captions, sizes):
                    matas[img_path] = {"caption": caption, "resolution": list(size)}
                with open(info_cache_file, "w", encoding="utf-8") as f:
                    json.dump(matas, f, ensure_ascii=False, indent=2)
                logger.info(f"cache image info done for / 画像情報を出力しました : {info_cache_file}")

            # if sizes are not set, image size will be read in make_buckets
            return img_paths, captions, sizes

        logger.info("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[Tuple[ImageInfo, DreamBoothSubset]] = []
        for subset in subsets:
            if subset.num_repeats < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します"
                )
                continue

            img_paths, captions, sizes = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found / 画像が見つからないためサブセットを無視します"
                )
                continue

            if subset.is_reg:
                num_reg_images += subset.num_repeats * len(img_paths)
            else:
                num_train_images += subset.num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(img_path, subset.num_repeats, caption, subset.is_reg, img_path)
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        logger.info(f"{num_train_images} train images with repeating.")
        self.num_train_images = num_train_images

        logger.info(f"{num_reg_images} reg images.")
        if num_train_images < num_reg_images:
            logger.warning("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります")

        if num_reg_images == 0:
            logger.warning("no regularization images / 正則化画像が見つかりませんでした")
        else:
            # num_repeatsを計算する：どうせ大した数ではないのでループで処理する
            n = 0
            first_loop = True
            while n < num_train_images:
                for info, subset in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: bool,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, network_multiplier, debug_dataset)

        self.batch_size = batch_size

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                logger.warning(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1 / \
                        num_repeatsが1を下回っているためサブセットを無視します: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one / \
                        既にサブセットが登録されているため、重複した後発のサブセットを無視します"
                )
                continue

            # メタデータを読み込む
            if os.path.exists(subset.metadata_file):
                logger.info(f"loading existing metadata: {subset.metadata_file}")
                with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                raise ValueError(f"no metadata / メタデータファイルがありません: {subset.metadata_file}")

            if len(metadata) < 1:
                logger.warning(
                    f"ignore subset with '{subset.metadata_file}': no image entries found / 画像に関するデータが見つからないためサブセットを無視します"
                )
                continue

            tags_list = []
            for image_key, img_md in metadata.items():
                # path情報を作る
                abs_path = None

                # まず画像を優先して探す
                if os.path.exists(image_key):
                    abs_path = image_key
                elif os.path.exists(os.path.join(subset.image_dir, image_key)):
                    abs_path = os.path.join(subset.image_dir, image_key)
                else:
                    # わりといい加減だがいい方法が思いつかん
                    paths = glob_images(subset.image_dir, image_key)
                    if len(paths) > 0:
                        abs_path = paths[0]

                # なければnpzを探す
                if abs_path is None:
                    if os.path.exists(os.path.splitext(image_key)[0] + ".npz"):
                        abs_path = os.path.splitext(image_key)[0] + ".npz"
                    else:
                        npz_path = os.path.join(subset.image_dir, image_key + ".npz")
                        if os.path.exists(npz_path):
                            abs_path = npz_path

                assert abs_path is not None, f"no image / 画像がありません: {image_key}"

                caption = img_md.get("caption")
                tags = img_md.get("tags")
                if caption is None:
                    caption = tags  # could be multiline
                    tags = None

                if subset.enable_wildcard:
                    # tags must be single line
                    if tags is not None:
                        tags = tags.replace("\n", subset.caption_separator)

                    # add tags to each line of caption
                    if caption is not None and tags is not None:
                        caption = "\n".join(
                            [
                                f"{line}{subset.caption_separator}{tags}"
                                for line in caption.split("\n")
                                if line.strip() != ""
                            ]
                        )
                else:
                    # use as is
                    if tags is not None and len(tags) > 0:
                        caption = caption + subset.caption_separator + tags
                        tags_list.append(tags)

                if caption is None:
                    caption = ""

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path)
                image_info.image_size = img_md.get("train_resolution")

                if not subset.color_aug and not subset.random_crop:
                    # if npz exists, use them
                    image_info.latents_npz, image_info.latents_npz_flipped = self.image_key_to_npz_file(
                        subset, image_key
                    )

                self.register_image(image_info, subset)

            self.num_train_images += len(metadata) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata)
            self.subsets.append(subset)

        # check existence of all npz files
        use_npz_latents = all([not (subset.color_aug or subset.random_crop) for subset in self.subsets])
        if use_npz_latents:
            flip_aug_in_subset = False
            npz_any = False
            npz_all = True

            for image_info in self.image_data.values():
                subset = self.image_to_subset[image_info.image_key]

                has_npz = image_info.latents_npz is not None
                npz_any = npz_any or has_npz

                if subset.flip_aug:
                    has_npz = has_npz and image_info.latents_npz_flipped is not None
                    flip_aug_in_subset = True
                npz_all = npz_all and has_npz

                if npz_any and not npz_all:
                    break

            if not npz_any:
                use_npz_latents = False
                logger.warning("npz file does not exist. ignore npz files / npzファイルが見つからないためnpzファイルを無視します")
            elif not npz_all:
                use_npz_latents = False
                logger.warning("some of npz file does not exist. ignore npz files / いくつかのnpzファイルが見つからないためnpzファイルを無視します")
                if flip_aug_in_subset:
                    logger.warning("maybe no flipped files / 反転されたnpzファイルがないのかもしれません")
        # else:
        #   logger.info("npz files are not used with color_aug and/or random_crop / color_augまたはrandom_cropが指定されているためnpzファイルは使用されません")

        # check min/max bucket size
        sizes = set()
        resos = set()
        for image_info in self.image_data.values():
            if image_info.image_size is None:
                sizes = None  # not calculated
                break
            sizes.add(image_info.image_size[0])
            sizes.add(image_info.image_size[1])
            resos.add(tuple(image_info.image_size))

        if sizes is None:
            if use_npz_latents:
                use_npz_latents = False
                logger.warning(
                    "npz files exist, but no bucket info in metadata. ignore npz files / メタデータにbucket情報がないためnpzファイルを無視します"
                )

            assert (
                resolution is not None
            ), "if metadata doesn't have bucket info, resolution is required / メタデータにbucket情報がない場合はresolutionを指定してください"

            self.enable_bucket = enable_bucket
            if self.enable_bucket:
                self.min_bucket_reso = min_bucket_reso
                self.max_bucket_reso = max_bucket_reso
                self.bucket_reso_steps = bucket_reso_steps
                self.bucket_no_upscale = bucket_no_upscale
        else:
            if not enable_bucket:
                logger.info("metadata has bucket info, enable bucketing / メタデータにbucket情報があるためbucketを有効にします")
            logger.info("using bucket info in metadata / メタデータ内のbucket情報を使います")
            self.enable_bucket = True

            assert (
                not bucket_no_upscale
            ), "if metadata has bucket info, bucket reso is precalculated, so bucket_no_upscale cannot be used /\
                  メタデータ内にbucket情報がある場合はbucketの解像度は計算済みのため、bucket_no_upscaleは使えません"

            # bucket情報を初期化しておく、make_bucketsで再作成しない
            self.bucket_manager = BucketManager(False, None, None, None, None)
            self.bucket_manager.set_predefined_resos(resos)

        # npz情報をきれいにしておく
        if not use_npz_latents:
            for image_info in self.image_data.values():
                image_info.latents_npz = image_info.latents_npz_flipped = None

    def image_key_to_npz_file(self, subset: FineTuningSubset, image_key):
        base_name = os.path.splitext(image_key)[0]
        npz_file_norm = base_name + ".npz"

        if os.path.exists(npz_file_norm):
            # image_key is full path
            npz_file_flip = base_name + "_flip.npz"
            if not os.path.exists(npz_file_flip):
                npz_file_flip = None
            return npz_file_norm, npz_file_flip

        # if not full path, check image_dir. if image_dir is None, return None
        if subset.image_dir is None:
            return None, None

        # image_key is relative path
        npz_file_norm = os.path.join(subset.image_dir, image_key + ".npz")
        npz_file_flip = os.path.join(subset.image_dir, image_key + "_flip.npz")

        if not os.path.exists(npz_file_norm):
            npz_file_norm = None
            npz_file_flip = None
        elif not os.path.exists(npz_file_flip):
            npz_file_flip = None

        return npz_file_norm, npz_file_flip


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset: float,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, network_multiplier, debug_dataset)

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_cropはControlNetDatasetではサポートされていません"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            batch_size,
            tokenizer,
            max_token_length,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            debug_dataset,
        )

        # config_util等から参照される値をいれておく（若干微妙なのでなんとかしたい）
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(
                os.path.splitext(ctrl_img_path)[0]
            )  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / 制御用画像が見つかりませんでした: {missing_imgs}"
        assert (
            len(extra_imgs) == 0
        ), f"extra conditioning data for {len(extra_imgs)} images / 余分な制御用画像があります: {extra_imgs}"

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        return self.dreambooth_dataset_delegate.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]  # noqa: F841
            flipped = example["flippeds"][i]
            cond_img = load_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / 画像サイズが合いません: {image_info.absolute_path}"
                cond_img = cv2.resize(
                    cond_img, image_info.resized_size, interpolation=cv2.INTER_AREA
                )  # INTER_AREAでやりたいのでcv2でリサイズ

                # TODO support random crop
                # 現在サポートしているcropはrandomではなく中央のみ
                h, w = target_size_hw
                ct = (cond_img.shape[0] - h) // 2
                cl = (cond_img.shape[1] - w) // 2
                cond_img = cond_img[ct : ct + h, cl : cl + w]
            else:
                # assert (
                #     cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                # ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"
                # resize to target
                if cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
                    cond_img = cv2.resize(
                        cond_img, (int(target_size_hw[1]), int(target_size_hw[0])), interpolation=cv2.INTER_LANCZOS4
                    )

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = ops.stack(conditioning_images).float()

        return example


# behave as Dataset mock
class DatasetGroup:
    # notes: implements `torch.utils.data.ConcatDataset`-like datasets here

    def __init__(self, datasets: Sequence[Union[DreamBoothDataset, FineTuningDataset]]):
        self.datasets = datasets
        self.cumulative_sizes = self.cumsum(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)  # noqa: E741
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(tokenizers, text_encoders, weight_dtype, cache_to_disk, is_main_process)

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self) -> bool:
        return all([dataset.is_text_encoder_output_cacheable() for dataset in self.datasets])

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_resoはWxHなので注意

    if not os.path.exists(npz_path):
        return False

    npz = np.load(npz_path)
    if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
        return False
    if npz["latents"].shape[1:3] != expected_latents_size:
        return False

    if flip_aug:
        if "latents_flipped" not in npz:
            return False
        if npz["latents_flipped"].shape[1:3] != expected_latents_size:
            return False

    return True


# 戻り値は、latents_tensor, (original_size width, original_size height), (crop left, crop top)
def load_latents_from_disk(
    npz_path,
) -> Tuple[Optional[ms.Tensor], Optional[List[int]], Optional[List[int]], Optional[ms.Tensor]]:
    npz = np.load(npz_path)
    if "latents" not in npz:
        raise ValueError(f"error: npz is old format. please re-generate {npz_path}")

    latents = npz["latents"]
    original_size = npz["original_size"].tolist()
    crop_ltrb = npz["crop_ltrb"].tolist()
    flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
    return latents, original_size, crop_ltrb, flipped_latents


def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None):
    kwargs = {}
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped"] = flipped_latents_tensor.float().numpy()
    np.savez(
        npz_path,
        latents=latents_tensor.float().numpy(),
        original_size=np.array(original_size),
        crop_ltrb=np.array(crop_ltrb),
        **kwargs,
    )


def debug_dataset(train_dataset, show_input_ids=False):
    logger.info(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
    logger.info("`S` for next step, `E` for next epoch no. , Escape for exit. / Sキーで次のステップ、Eキーで次のエポック、Escキーで中断、終了します")

    epoch = 1
    while True:
        logger.info(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            logger.info(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                logger.info(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, iid, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    example["input_ids"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                logger.info(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", \
                        original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )
                if "network_multipliers" in example:
                    print(f"network multiplier: {example['network_multipliers'][j]}")

                if show_input_ids:
                    logger.info(f"input ids: {iid}")
                    if "input_ids2" in example:
                        logger.info(f"input ids2: {example['input_ids2'][j]}")
                if example["images"] is not None:
                    im = example["images"][j]
                    logger.info(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example:
                        cond_img = example["conditioning_images"][j]
                        logger.info(f"conditioning image size: {cond_img.size()}")
                        cond_img = ((cond_img.numpy() + 1.0) * 127.5).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # 重複を排除
    img_paths.sort()
    return img_paths


def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths


class MinimalDataset(BaseDataset):
    def __init__(self, tokenizer, max_token_length, resolution, network_multiplier, debug_dataset=False):
        super().__init__(tokenizer, max_token_length, resolution, network_multiplier, debug_dataset)

        self.num_train_images = 0  # update in subclass
        self.num_reg_images = 0  # update in subclass
        self.datasets = [self]
        self.batch_size = 1  # update in subclass

        self.subsets = [self]
        self.num_repeats = 1  # update in subclass if needed
        self.img_count = 1  # update in subclass if needed
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"  # for metadata

    def verify_bucket_reso_steps(self, min_steps: int):
        pass

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    # override to avoid shuffling buckets
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        r"""
        The subclass may have image_data for debug_dataset, which is a dict of ImageInfo objects.

        Returns: example like this:

            for i in range(batch_size):
                image_key = ...  # whatever hashable
                image_keys.append(image_key)

                image = ...  # PIL Image
                img_tensor = self.image_transforms(img)
                images.append(img_tensor)

                caption = ...  # str
                input_ids = self.get_input_ids(caption)
                input_ids_list.append(input_ids)

                captions.append(caption)

            images = torch.stack(images, dim=0)
            input_ids_list = torch.stack(input_ids_list, dim=0)
            example = {
                "images": images,
                "input_ids": input_ids_list,
                "captions": captions,   # for debug_dataset
                "latents": None,
                "image_keys": image_keys,   # for debug_dataset
                "loss_weights": torch.ones(batch_size, dtype=torch.float32),
            }
            return example
        """
        raise NotImplementedError


def load_arbitrary_dataset(args, tokenizer) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(
        tokenizer, args.max_token_length, args.resolution, args.debug_dataset
    )
    return train_dataset_group


def load_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    img = np.array(image, np.uint8)
    return img


# 画像を読み込む。戻り値はnumpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)
def trim_and_resize_if_required(
    random_crop: bool, image: Image.Image, reso, resized_size: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        # リサイズする
        image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)  # INTER_AREAでやりたいのでcv2でリサイズ

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random cropの場合のcropされた値をどうcrop left/topに反映するべきか全くアイデアがない
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    assert (
        image.shape[0] == reso[1] and image.shape[1] == reso[0]
    ), f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


def load_text_encoder_outputs_from_disk(npz_path):
    with np.load(npz_path) as f:
        hidden_state1 = ms.Tensor.from_numpy(f["hidden_state1"])
        hidden_state2 = ms.Tensor.from_numpy(f["hidden_state2"]) if "hidden_state2" in f else None
        pool2 = ms.Tensor.from_numpy(f["pool2"]) if "pool2" in f else None
    return hidden_state1, hidden_state2, pool2


# endregion

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
        )
    except BaseException:
        return "(unknown)"


def replace_unet_modules(unet, flash_attn):
    if flash_attn:
        logger.info("Enable mindspore flash attention for U-Net")
        unet.set_use_memory_efficient_attention(True, False)


# endregion


# region arguments


# this metadata is referred from train_network and various scripts, so we wrote here
SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


def build_minimum_network_metadata(
    v2: Optional[bool],
    base_model: Optional[str],
    network_module: str,
    network_dim: str,
    network_alpha: str,
    network_args: Optional[dict],
):
    # old LoRA doesn't have base_model
    metadata = {
        SS_METADATA_KEY_NETWORK_MODULE: network_module,
        SS_METADATA_KEY_NETWORK_DIM: network_dim,
        SS_METADATA_KEY_NETWORK_ALPHA: network_alpha,
    }
    if v2 is not None:
        metadata[SS_METADATA_KEY_V2] = v2
    if base_model is not None:
        metadata[SS_METADATA_KEY_BASE_MODEL_VERSION] = base_model
    if network_args is not None:
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(network_args)
    return metadata


def get_sai_model_spec(
    state_dict: dict,
    args: argparse.Namespace,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    is_stable_diffusion_ckpt: Optional[bool] = None,  # None for TI and LoRA
):
    timestamp = time.time()

    v2 = args.v2
    v_parameterization = args.v_parameterization
    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    metadata = sai_model_spec.build_metadata(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,  # None or int
    )
    return metadata


def add_sd_models_arguments(parser: argparse.ArgumentParser):
    # for pretrained models
    parser.add_argument(
        "--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む"
    )
    parser.add_argument(
        "--v_parameterization",
        action="store_true",
        help="enable v-parameterization training / v-parameterization学習を有効にする",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="AdamW",
        help="Optimizer to use / オプティマイザの種類: support AdamW (default) only",
    )

    # backward compatibility
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch) / Lionオプティマイザを使う（ lion-pytorch のインストールが必要）",
    )

    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない",
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',
    )

    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / \
            学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler (default is 0) / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to output trained model / 学習後のモデル出力先ディレクトリ"
    )
    parser.add_argument(
        "--output_name", type=str, default=None, help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名"
    )
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files / huggingfaceにアップロードするファイルのパス",
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private) / \
            huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）",
    )
    parser.add_argument(
        "--save_state_to_huggingface", action="store_true", help="save state to huggingface / huggingfaceにstateを保存する"
    )
    parser.add_argument(
        "--resume_from_huggingface",
        action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する\
            (例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously / huggingfaceに非同期でアップロードする",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp32", "fp16", "bf16"],
        help="precision in saving / 保存時に精度を変更して保存する",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps / 学習中のモデルを指定ステップごとに保存する",
    )
    parser.add_argument(
        "--save_n_epoch_ratio",
        type=int,
        default=None,
        help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total) / 学習中のモデルを指定のエポック割合で保存する（たとえば5を指定すると最低5個のファイルが保存される）",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints) / 指定エポックごとにモデルを保存するとき最大Nエポック保存する（古いチェックポイントは削除する）",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する（--save_last_n_epochsの指定を上書きする）",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) / 指定ステップごとにモデルを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する）",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) / \
            指定ステップごとにstateを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する。--save_last_n_stepsを上書きする）",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model /\
              optimizerなど学習状態も含めたstateをモデル保存時に追加で保存する",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end / optimizerなど学習状態も含めたstateを学習完了時に保存する",
    )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        choices=[None, 150, 225],
        help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）",
    )
    parser.add_argument(
        "--torch_compile", action="store_true", help="use torch.compile (requires PyTorch 2.0) / torch.compile を使う"
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="inductor",
        # available backends:
        # https://github.com/huggingface/accelerate/blob/d1abd59114ada8ba673e1214218cb2878c13b82d/src/accelerate/utils/dataclasses.py#L376-L388C5
        # https://pytorch.org/docs/stable/torch.compiler.html
        choices=[
            "eager",
            "aot_eager",
            "inductor",
            "aot_ts_nvfuser",
            "nvprims_nvfuser",
            "cudagraphs",
            "ofi",
            "fx2trt",
            "onnxrt",
        ],
        help="dynamo backend type (default is inductor) / dynamoのbackendの種類（デフォルトは inductor）",
    )
    parser.add_argument(
        "--flash_attn", action="store_true", help="use flash_attn for CrossAttention / CrossAttentionにflash_attnを使う"
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ",
    )

    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / \
            DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / \
            DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="enable gradient checkpointing / grandient checkpointingを有効にする",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度",
    )
    parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument(
        "--full_bf16", action="store_true", help="bf16 training including gradients / 勾配も含めてbf16で学習する"
    )  # TODO move to SDXL training, because it is not supported by SD1/2
    parser.add_argument("--fp8_base", action="store_true", help="use fp8 for base model / base modelにfp8を使う")

    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=None,
        help="DDP timeout (min, None for default of accelerate) / DDPのタイムアウト（分、Noneでaccelerateのデフォルト）",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action="store_true",
        help="enable gradient_as_bucket_view for DDP / DDPでgradient_as_bucket_viewを有効にする",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action="store_true",
        help="enable static_graph for DDP / DDPでstatic_graphを有効にする",
    )
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is script-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session / wandb ログに表示される特定の実行の名前",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging / ログ出力に使用するtrackerの設定ファイルのパス",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）",
    )

    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）",
    )
    parser.add_argument(
        "--noise_offset_random_strength",
        action="store_true",
        help="use random strength between 0~noise_offset for noise offset. / noise offsetにおいて、0からnoise_offsetの間でランダムな強度を使用します。",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended) / \
            Multires noiseを有効にしてこのイテレーション数を設定する（有効にする場合は6-10程度を推奨）",
    )
    parser.add_argument(
        "--ip_noise_gamma",
        type=float,
        default=None,
        help="enable input perturbation noise. used for regularization. recommended value: around 0.1 (from arxiv.org/abs/2301.11706) "
        + "/  input perturbation noiseを有効にする。正則化に使用される。推奨値: 0.1程度 (arxiv.org/abs/2301.11706 より)",
    )
    parser.add_argument(
        "--ip_noise_gamma_random_strength",
        action="store_true",
        help="Use random strength between 0~ip_noise_gamma for input perturbation noise."
        + "/ input perturbation noiseにおいて、0からip_noise_gammaの間でランダムな強度を使用します。",
    )
    # parser.add_argument(
    #     "--perlin_noise",
    #     type=int,
    #     default=None,
    #     help="enable perlin noise and set the octaves / perlin noiseを有効にしてoctavesをこの値に設定する",
    # )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations) / \
            Multires noiseのdiscount値を設定する（--multires_noise_iterations指定時のみ有効）",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default) / \
            latentの平均値の絶対値 * この値をnoise_offsetに加算する（Noneの場合は無効、デフォルト）",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="fix noise scheduler betas to enforce zero terminal SNR / noise schedulerのbetasを修正して、zero terminal SNRを強制する",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for U-Net training (0~999, default is 0) / U-Net学習時のtime stepの最小値を設定する（0~999で指定、省略時はデフォルト値(0)） ",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for U-Net training (1~1000, default is 1000) / U-Net学習時のtime stepの最大値を設定する（1~1000で指定、省略時はデフォルト値(1000)）",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber", "smooth_l1"],
        help="The type of loss function to use (L2, Huber, or smooth L1), default is L2 / 使用する損失関数の種類（L2、Huber、またはsmooth L1）、デフォルトはL2",
    )
    parser.add_argument(
        "--huber_schedule",
        type=str,
        default="snr",
        choices=["constant", "exponential", "snr"],
        help="The scheduling method for Huber loss (constant, exponential, or SNR-based). Only used when loss_type is 'huber' or 'smooth_l1'. default is snr"
        + " / Huber損失のスケジューリング方法（constant、exponential、またはSNRベース）。loss_typeが'huber'または'smooth_l1'の場合に有効、デフォルトは snr",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.1,
        help="The huber loss parameter. Only used if one of the huber loss modes (huber or smooth l1) is selected with loss_type. \
            default is 0.1 / Huber損失のパラメータ。loss_typeがhuberまたはsmooth l1の場合に有効。デフォルトは0.1",
    )

    parser.add_argument(
        "--lowram",
        action="store_true",
        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle) / \
            メインメモリが少ない環境向け最適化を有効にする。たとえばVRAMにモデルを読み込む等（ColabやKaggleなどRAMに比べてVRAMが多い環境向け）",
    )
    parser.add_argument(
        "--highvram",
        action="store_true",
        help="disable low VRAM optimization. e.g. do not clear CUDA cache after each latent caching (for machines which have bigger VRAM) "
        + "/ VRAMが少ない環境向け最適化を無効にする。たとえば各latentのキャッシュ後のCUDAキャッシュクリアを行わない等（VRAMが多い環境向け）",
    )

    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する",
    )
    parser.add_argument(
        "--sample_at_first", action="store_true", help="generate sample images before training / 学習前にサンプル出力する"
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル",
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help="sampler (scheduler) type for sample images / サンプル出力時のサンプラー（スケジューラ）の種類",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter / ハイパーパラメータを引数ではなく.tomlファイルで渡す",
    )
    parser.add_argument(
        "--output_config", action="store_true", help="output command line args to given .toml file / 引数を.tomlファイルに出力する"
    )

    # SAI Model spec
    parser.add_argument(
        "--metadata_title",
        type=str,
        default=None,
        help="title for model metadata (default is output_name) / メタデータに書き込まれるモデルタイトル、省略時はoutput_name",
    )
    parser.add_argument(
        "--metadata_author",
        type=str,
        default=None,
        help="author name for model metadata / メタデータに書き込まれるモデル作者名",
    )
    parser.add_argument(
        "--metadata_description",
        type=str,
        default=None,
        help="description for model metadata / メタデータに書き込まれるモデル説明",
    )
    parser.add_argument(
        "--metadata_license",
        type=str,
        default=None,
        help="license for model metadata / メタデータに書き込まれるモデルライセンス",
    )
    parser.add_argument(
        "--metadata_tags",
        type=str,
        default=None,
        help="tags for model metadata, separated by comma / メタデータに書き込まれるモデルタグ、カンマ区切り",
    )

    if support_dreambooth:
        # DreamBooth training
        parser.add_argument(
            "--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み"
        )


def add_masked_loss_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory / 条件付けデータのディレクトリ",
    )
    parser.add_argument(
        "--masked_loss",
        action="store_true",
        help="apply mask for calculating loss. conditioning_data_dir is required for dataset. / 損失計算時にマスクを適用する。datasetにはconditioning_data_dirが必要",
    )


# # verify command line args for training
def verify_command_line_training_args(args: argparse.Namespace):
    # if wandb is enabled, the command line is exposed to the public
    # check whether sensitive options are included in the command line arguments
    # if so, warn or inform the user to move them to the configuration file
    # wandbが有効な場合、コマンドラインが公開される
    # 学習用のコマンドライン引数に敏感なオプションが含まれているかどうかを確認し、
    # 含まれている場合は設定ファイルに移動するようにユーザーに警告または通知する

    wandb_enabled = args.log_with is not None and args.log_with != "tensorboard"  # "all" or "wandb"
    if not wandb_enabled:
        return

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]

    for arg in sensitive_args:
        if getattr(args, arg, None) is not None:
            logger.warning(
                f"wandb is enabled, but option `{arg}` is included in the command line. \
                    Because the command line is exposed to the public, it is recommended to move it to the `.toml` file."
                + f" / wandbが有効で、かつオプション `{arg}` がコマンドラインに含まれています。\
                    コマンドラインは公開されるため、`.toml`ファイルに移動することをお勧めします。"
            )

    # if path is absolute, it may include sensitive information
    for arg in sensitive_path_args:
        if getattr(args, arg, None) is not None and os.path.isabs(getattr(args, arg)):
            logger.info(
                f"wandb is enabled, but option `{arg}` is included in the command line and it is an absolute path. \
                    Because the command line is exposed to the public, it is recommended to move it to the `.toml` file or use relative path."
                + f" / wandbが有効で、かつオプション `{arg}` がコマンドラインに含まれており、絶対パスです。\
                    コマンドラインは公開されるため、`.toml`ファイルに移動するか、相対パスを使用することをお勧めします。"
            )

    if getattr(args, "config_file", None) is not None:
        logger.info(
            "wandb is enabled, but option `config_file` is included in the command line. \
                Because the command line is exposed to the public, please be careful about the information included in the path."
            + " / wandbが有効で、かつオプション `config_file` がコマンドラインに含まれています。\
                コマンドラインは公開されるため、パスに含まれる情報にご注意ください。"
        )

    # other sensitive options
    if args.huggingface_repo_id is not None and args.huggingface_repo_visibility != "public":
        logger.info(
            "wandb is enabled, but option huggingface_repo_id is included in the command line and huggingface_repo_visibility is not 'public'. \
                Because the command line is exposed to the public, it is recommended to move it to the `.toml` file."
            + " / wandbが有効で、かつオプション huggingface_repo_id がコマンドラインに含まれており、huggingface_repo_visibility が 'public' ではありません。\
                コマンドラインは公開されるため、`.toml`ファイルに移動することをお勧めします。"
        )


def verify_training_args(args: argparse.Namespace):
    r"""
    Verify training arguments. Also reflect highvram option to global variable
    学習用引数を検証する。あわせて highvram オプションの指定をグローバル変数に反映する
    """
    if args.highvram:
        print("highvram is enabled / highvramが有効です")
        global HIGH_VRAM
        HIGH_VRAM = True

    if args.v_parameterization and not args.v2:
        logger.warning(
            "v_parameterization should be with v2 not v1 or sdxl / v1やsdxlでv_parameterizationを使用することは想定されていません"
        )
    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

    if args.cache_latents_to_disk and not args.cache_latents:
        args.cache_latents = True
        logger.warning(
            "cache_latents_to_disk is enabled, so cache_latents is also enabled / cache_latents_to_diskが有効なため、cache_latentsを有効にします"
        )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset / adaptive_noise_scaleを使用するにはnoise_offsetが必要です")

    if args.scale_v_pred_loss_like_noise_pred and not args.v_parameterization:
        raise ValueError(
            "scale_v_pred_loss_like_noise_pred can be enabled only with v_parameterization / \
                scale_v_pred_loss_like_noise_predはv_parameterizationが有効なときのみ有効にできます"
        )

    if args.v_pred_like_loss and args.v_parameterization:
        raise ValueError(
            "v_pred_like_loss cannot be enabled with v_parameterization / v_pred_like_lossはv_parameterizationが有効なときには有効にできません"
        )

    if args.zero_terminal_snr and not args.v_parameterization:
        logger.warning(
            "zero_terminal_snr is enabled, but v_parameterization is not enabled. training will be unexpected"
            + " / zero_terminal_snrが有効ですが、v_parameterizationが有効ではありません。学習結果は想定外になる可能性があります"
        )

    if args.sample_every_n_epochs is not None and args.sample_every_n_epochs <= 0:
        logger.warning(
            "sample_every_n_epochs is less than or equal to 0, so it will be disabled / sample_every_n_epochsに0以下の値が指定されたため無効になります"
        )
        args.sample_every_n_epochs = None

    if args.sample_every_n_steps is not None and args.sample_every_n_steps <= 0:
        logger.warning(
            "sample_every_n_steps is less than or equal to 0, so it will be disabled / sample_every_n_stepsに0以下の値が指定されたため無効になります"
        )
        args.sample_every_n_steps = None


def add_dataset_arguments(
    parser: argparse.ArgumentParser, support_dreambooth: bool, support_caption: bool, support_caption_dropout: bool
):
    # dataset common
    parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument(
        "--cache_info",
        action="store_true",
        help="cache meta information (caption and image size) for faster dataset loading. only available for DreamBooth"
        + " / メタ情報（キャプションとサイズ）をキャッシュしてデータセット読み込みを高速化する。DreamBooth方式のみ有効",
    )
    parser.add_argument(
        "--shuffle_caption", action="store_true", help="shuffle separated caption / 区切られたcaptionの各要素をshuffleする"
    )
    parser.add_argument("--caption_separator", type=str, default=",", help="separator for caption / captionの区切り文字")
    parser.add_argument(
        "--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子"
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）",
    )
    parser.add_argument(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens (token means comma separated strings) / \
            captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す（トークンはカンマ区切りの各部分を意味する）",
    )
    parser.add_argument(
        "--keep_tokens_separator",
        type=str,
        default="",
        help="A custom separator to divide the caption into fixed and flexible parts. Tokens before this separator will not be shuffled. \
            If not specified, '--keep_tokens' will be used to determine the fixed number of tokens."
        + " / captionを固定部分と可変部分に分けるためのカスタム区切り文字。この区切り文字より前のトークンはシャッフルされない。指定しない場合、\
            '--keep_tokens'が固定部分のトークン数として使用される。",
    )
    parser.add_argument(
        "--secondary_separator",
        type=str,
        default=None,
        help="a secondary separator for caption. This separator is replaced to caption_separator after dropping/shuffling caption"
        + " / captionのセカンダリ区切り文字。この区切り文字はcaptionのドロップやシャッフル後にcaption_separatorに置き換えられる",
    )
    parser.add_argument(
        "--enable_wildcard",
        action="store_true",
        help="enable wildcard for caption (e.g. '{image|picture|rendition}') / captionのワイルドカードを有効にする（例：'{image|picture|rendition}'）",
    )
    parser.add_argument(
        "--caption_prefix",
        type=str,
        default=None,
        help="prefix for caption text / captionのテキストの先頭に付ける文字列",
    )
    parser.add_argument(
        "--caption_suffix",
        type=str,
        default=None,
        help="suffix for caption text / captionのテキストの末尾に付ける文字列",
    )
    parser.add_argument(
        "--color_aug", action="store_true", help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする"
    )
    parser.add_argument(
        "--flip_aug", action="store_true", help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする"
    )
    parser.add_argument(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) / 学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="enable random crop (for style training in face-centered crop augmentation) / ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）",
    )
    parser.add_argument(
        "--debug_dataset",
        action="store_true",
        help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="cache latents to main memory to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをメインメモリにcacheする（augmentationは使用不可） ",
    )
    parser.add_argument(
        "--vae_batch_size", type=int, default=1, help="batch size for caching latents / latentのcache時のバッチサイズ"
    )
    parser.add_argument(
        "--cache_latents_to_disk",
        action="store_true",
        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをディスクにcacheする（augmentationは使用不可）",
    )
    parser.add_argument(
        "--enable_bucket",
        action="store_true",
        help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする",
    )
    parser.add_argument(
        "--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度"
    )
    parser.add_argument(
        "--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最大解像度"
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します",
    )

    parser.add_argument(
        "--token_warmup_min",
        type=int,
        default=1,
        help="start learning at N tags (token means comma separated strinfloatgs) / タグ数をN個から増やしながら学習する",
    )
    parser.add_argument(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps (or N*max_train_steps if N<1) / N（N<1ならN*max_train_steps）ステップでタグ長が最大になる。デフォルトは0（最初から最大）",
    )

    parser.add_argument(
        "--dataset_class",
        type=str,
        default=None,
        help="dataset class for arbitrary dataset (package.module.Class) / 任意のデータセットを用いるときのクラス名 (package.module.Class)",
    )

    if support_caption_dropout:
        # Textual Inversion はcaptionのdropoutをsupportしない
        # いわゆるtensorのDropoutと紛らわしいのでprefixにcaptionを付けておく　every_n_epochsは他と平仄を合わせてdefault Noneに
        parser.add_argument(
            "--caption_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout caption(0.0~1.0) / captionをdropoutする割合",
        )
        parser.add_argument(
            "--caption_dropout_every_n_epochs",
            type=int,
            default=0,
            help="Dropout all captions every N epochs / captionを指定エポックごとにdropoutする",
        )
        parser.add_argument(
            "--caption_tag_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout comma separated tokens(0.0~1.0) / カンマ区切りのタグをdropoutする割合",
        )

    if support_dreambooth:
        # DreamBooth dataset
        parser.add_argument(
            "--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ"
        )

    if support_caption:
        # caption dataset
        parser.add_argument(
            "--in_json", type=str, default=None, help="json metadata for dataset / データセットのmetadataのjsonファイル"
        )
        parser.add_argument(
            "--dataset_repeats",
            type=int,
            default=1,
            help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数",
        )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）",
    )


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file

    if args.output_config:
        # check if config file exists
        if os.path.exists(config_path):
            logger.error(f"Config file already exists. Aborting... / 出力先の設定ファイルが既に存在します: {config_path}")
            exit(1)

        # convert args to dictionary
        args_dict = vars(args)

        # remove unnecessary keys
        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        # get default args from parser
        default_args = vars(parser.parse_args([]))

        # remove default values: cannot use args_dict.items directly because it will be changed during iteration
        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        # convert Path to str in dictionary
        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        # convert to toml and output to file
        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        logger.info(f"Saved config file / 設定ファイルを保存しました: {config_path}")
        exit(0)

    if not os.path.exists(config_path):
        logger.info(f"{config_path} not found.")
        exit(1)

    logger.info(f"Loading settings from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)

    # combine all sections into one
    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]
    logger.info(args.config_file)

    return args


# endregion

# # region utils


def get_optimizer(args, learning_rate, trainable_params):
    optimizer_type = args.optimizer_type
    if args.use_8bit_adam:
        assert (
            not args.use_lion_optimizer
        ), "both option use_8bit_adam and use_lion_optimizer are specified / use_8bit_adamとuse_lion_optimizerの両方のオプションが指定されています"
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_8bit_adam and optimizer_type are specified / use_8bit_adamとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "AdamW8bit"

    elif args.use_lion_optimizer:
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_lion_optimizer and optimizer_type are specified / use_lion_optimizerとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    # 引数を分解する
    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
    if "betas" in optimizer_kwargs:
        optimizer_kwargs["beta1"], optimizer_kwargs["beta2"] = optimizer_kwargs["betas"]
        del optimizer_kwargs["betas"]
    optimizer = None

    if optimizer_type != "AdamW".lower():
        raise NotImplementedError("now support AdamW only")
    else:
        logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = nn.AdamWeightDecay
        optimizer = optimizer_class(trainable_params, learning_rate=learning_rate, **optimizer_kwargs)

    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    return optimizer_name, optimizer_args, optimizer


# # Modified version of get_scheduler() function from diffusers.optimizer.get_scheduler
# # Add some checking and features to the original function.


def get_scheduler_fix(args, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    name = args.lr_scheduler
    num_warmup_steps: Optional[int] = args.lr_warmup_steps
    num_training_steps = args.max_train_steps * num_processes  # * args.gradient_accumulation_steps
    num_cycles = args.lr_scheduler_num_cycles
    power = args.lr_scheduler_power
    lr = args.learning_rate

    lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
    if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
        for arg in args.lr_scheduler_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    # usinglr_scheduler from other library
    if args.lr_scheduler_type:
        NotImplementedError("now not support any lr_scheduler from other library")

    if name.startswith("adafactor"):
        NotImplementedError

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # All schedulers require `num_training_steps` (based on mindone.diffusers implements)
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(lr, num_training_steps=num_training_steps))

    if name == SchedulerType.PIECEWISE_CONSTANT:
        assert (
            "step_rules" in lr_scheduler_kwargs
        ), f"{name} requires `num_training_steps`, please provide that argument in args.lr_scheduler_args."
        return schedule_func(
            lr, step_rules=lr_scheduler_kwargs["step_rules"], num_training_steps=num_training_steps
        )  # step_rules and last_epoch are given as kwargs

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(lr, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            lr,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )

    if name == SchedulerType.POLYNOMIAL:
        assert lr_scheduler_kwargs == {} or (len(lr_scheduler_kwargs) == 1 and "lr_end" in lr_scheduler_kwargs)
        return schedule_func(
            lr,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            **lr_scheduler_kwargs,
        )

    return schedule_func(lr, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    # backward compatibility
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    # assert args.resolution is not None, f"resolution is required / resolution（解像度）を指定してください"
    if args.resolution is not None:
        args.resolution = tuple([int(r) for r in args.resolution.split(",")])
        if len(args.resolution) == 1:
            args.resolution = (args.resolution[0], args.resolution[0])
        assert (
            len(args.resolution) == 2
        ), f"resolution must be 'size' or 'width,height' / resolution（解像度）は'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(",")])
        assert (
            len(args.face_crop_aug_range) == 2 and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1]
        ), f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            logger.warning(
                "latents in npz is ignored when color_aug or random_crop is True \
                    / color_augまたはrandom_cropを有効にした場合、npzファイルのlatentsは無視されます"
            )


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = ms.float16
    elif args.save_precision == "bf16":
        save_dtype = ms.bfloat16
    elif args.save_precision == "float":
        save_dtype = ms.float32

    return weight_dtype, save_dtype


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection, last_hidden_state: ms.Tensor, input_ids: ms.Tensor, eos_token_id: int
):
    r"""
    workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
    instead of the hidden states for the EOS token
    If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

    Original code from CLIP's pooling function:

    \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
    \# take features from the eot embedding (eot_token is the highest number in each sequence)
    \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]
    """

    # input_ids: b*n,77
    # find index for EOS token

    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # Use argmax to find the last index of the EOS token for each element in the batch
    eos_token_index = ops.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine

    # get hidden states for EOS token
    pooled_output = last_hidden_state[ops.arange(last_hidden_state.shape[0]), eos_token_index]

    # apply projection: projection may be of different dtype than last_hidden_state
    pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)

    return pooled_output


def get_hidden_states_sdxl(
    max_token_length: int,
    input_ids1: ms.Tensor,
    input_ids2: ms.Tensor,
    tokenizer1_model_max_length,
    tokenizer2_model_max_length,
    tokenizer2_eos_token_id,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
    weight_dtype: Optional[str] = None,
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.shape[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1_model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2_model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=False)
    hidden_states1 = enc_out[-1][-2]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=False)
    hidden_states2 = enc_out[-1][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    pool2 = pool_workaround(text_encoder2, enc_out[1], input_ids2, tokenizer2_eos_token_id)

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1_model_max_length):
            states_list.append(hidden_states1[:, i : i + tokenizer1_model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = ops.cat(states_list, axis=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2_model_max_length):
            chunk = hidden_states2[:, i : i + tokenizer2_model_max_length - 2]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
        hidden_states2 = ops.cat(states_list, axis=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def default_if_none(value, default):
    return default if value is None else value


def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name, step_no) + ext


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
    # save_every_n_steps=10, save_last_n_steps=30の場合、50step目には30step分残し、10step目を削除する
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, network, epoch_no):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    ms.save_checkpoint(network, state_dir)
    if args.save_state_to_huggingface:
        logger.warning("not support uploading state to huggingface yet")

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, network, step_no):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    ms.save_checkpoint(network, state_dir)
    if args.save_state_to_huggingface:
        logger.warning("not support uploading state to huggingface yet")

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, model):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    ms.save_checkpoint(model, state_dir)

    if args.save_state_to_huggingface:
        logger.warning("uploading last state to huggingface not supported yet")


def get_timesteps_and_huber_c(args, min_timestep, max_timestep, noise_scheduler, b_size):
    # TODO: if a huber loss is selected, it will use constant timesteps for each batch
    # as. In the future there may be a smarter way

    if args.loss_type == "huber" or args.loss_type == "smooth_l1":
        timesteps = ops.randint(min_timestep, max_timestep, (1,))
        timestep = timesteps.item()

        if args.huber_schedule == "exponential":
            alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
            huber_c = math.exp(-alpha * timestep)
        elif args.huber_schedule == "snr":
            alphas_cumprod = noise_scheduler.alphas_cumprod[timestep]
            sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
            huber_c = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        elif args.huber_schedule == "constant":
            huber_c = args.huber_c
        else:
            raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

        timesteps = timesteps.repeat(b_size)
    elif args.loss_type == "l2":
        timesteps = ops.randint(min_timestep, max_timestep, (b_size,))
        huber_c = 1  # may be anything, as it's not used
    else:
        raise NotImplementedError(f"Unknown loss type {args.loss_type}")
    timesteps = timesteps.long()

    return timesteps, huber_c


# notes: `dict.key` is not supported in graph mode.
# The `args` dict is wrapped by AttrJitWrapper in trainstep while the noise_scheduler.config dose not.
# so we directly pass the two necessary args of noise_scheduler.config here, `max_timestep` and `min_timestep`
def get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, max_timestep=1000, min_timestep=0):
    # Sample noise that we'll add to the latents
    noise = ops.randn_like(latents, dtype=latents.dtype)
    if args.noise_offset:
        if args.noise_offset_random_strength:
            noise_offset = ops.rand(1, dtype=noise.dtype) * args.noise_offset
        else:
            noise_offset = args.noise_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)
    if args.multires_noise_iterations:
        noise = custom_train_functions.pyramid_noise_like(
            noise, args.multires_noise_iterations, args.multires_noise_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]

    timesteps, huber_c = get_timesteps_and_huber_c(args, min_timestep, max_timestep, noise_scheduler, b_size)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        if args.ip_noise_gamma_random_strength:
            strength = ops.rand(1, dtype=latents.dtype) * args.ip_noise_gamma
        else:
            strength = args.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * ops.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return noise, noisy_latents, timesteps, huber_c


# NOTE: if you're using the scheduled version, huber_c has to depend on the timesteps already
def conditional_loss(
    model_pred: ms.Tensor, target: ms.Tensor, reduction: str = "mean", loss_type: str = "l2", huber_c: float = 0.1
):
    if loss_type == "l2":
        loss = ops.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        loss = 2 * huber_c * (ops.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = ops.mean(loss)
        elif reduction == "sum":
            loss = ops.sum(loss)
    elif loss_type == "smooth_l1":
        loss = 2 * (ops.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = ops.mean(loss)
        elif reduction == "sum":
            loss = ops.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type {loss_type}")
    return loss


# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def get_my_scheduler(
    *,
    sample_sampler: str,
    v_parameterization: bool,
):
    sched_init_args = {}
    if sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sample_sampler == "lms" or sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sample_sampler == "euler" or sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sample_sampler == "euler_a" or sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sample_sampler == "dpmsolver" or sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sample_sampler
    elif sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sample_sampler == "dpm_2" or sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sample_sampler == "dpm_2_a" or sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # logger.info("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler


def sample_images(*args, **kwargs):
    logger.warning("not support train_util.sample_images yet")
    pass


# endregion


# collate_fn用 epoch,stepはmultiprocessing.Value
class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        dataset = self.dataset

        # set epoch and step
        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples


class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        return self.loss_total / len(self.loss_list)

"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion_safe/__init__.py."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import PIL
from PIL import Image

from ...utils import BaseOutput, _LazyModule


@dataclass
class SafetyConfig(object):
    WEAK = {
        "sld_warmup_steps": 15,
        "sld_guidance_scale": 20,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    }
    MEDIUM = {
        "sld_warmup_steps": 10,
        "sld_guidance_scale": 1000,
        "sld_threshold": 0.01,
        "sld_momentum_scale": 0.3,
        "sld_mom_beta": 0.4,
    }
    STRONG = {
        "sld_warmup_steps": 7,
        "sld_guidance_scale": 2000,
        "sld_threshold": 0.025,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    }
    MAX = {
        "sld_warmup_steps": 0,
        "sld_guidance_scale": 5000,
        "sld_threshold": 1.0,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    }


_additional_imports = {}
_import_structure = {}

_additional_imports.update({"SafetyConfig": SafetyConfig})

_import_structure.update(
    {
        "pipeline_output": ["StableDiffusionSafePipelineOutput"],
        "pipeline_stable_diffusion_safe": ["StableDiffusionPipelineSafe"],
        "safety_checker": ["StableDiffusionSafetyChecker"],
    }
)


if TYPE_CHECKING:
    from .pipeline_output import StableDiffusionSafePipelineOutput
    from .pipeline_stable_diffusion_safe import StableDiffusionPipelineSafe
    from .safety_checker import SafeStableDiffusionSafetyChecker

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)

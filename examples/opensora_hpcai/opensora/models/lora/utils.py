# reference to https://github.com/microsoft/LoRA

from typing import Literal

from mindspore import nn

from .layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Cell, bias: Literal["none", "all", "lora_only"] = "none") -> None:
    for n, p in model.parameters_and_names():
        if "lora_" not in n:
            p.requires_grad = False

    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.parameters_and_names():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.cells():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError

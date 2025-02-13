# reference to https://github.com/microsoft/LoRA
import math
from typing import Optional

from mindspore import Tensor, mint, ops
from mindspore.common import initializer as init


class LoRALayer:
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool) -> None:
        assert r > 0, f"LoRA layer rank dim must be greater than 0, but got {r}."
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = mint.nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = mint.nn.Identity()

        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(mint.nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ) -> None:
        assert r > 0, f"expected lora rank greater than 0, but got {r}"
        lora_alpha = lora_alpha if lora_alpha is not None else r
        dtype = kwargs.get("dtype", None)

        mint.nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = mint.nn.Linear(in_features, r, bias=False, dtype=dtype)
            self.lora_B = mint.nn.Linear(r, out_features, bias=False, dtype=dtype)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A.weight.set_data(
                init.initializer(
                    init.HeUniform(negative_slope=math.sqrt(5)), self.lora_A.weight.shape, self.lora_A.weight.dtype
                )
            )
            self.lora_B.weight.set_data(init.initializer(0.0, self.lora_B.weight.shape, self.lora_B.weight.dtype))

    def _set_train(self, mode: bool = True) -> None:
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    ops.assign(
                        self.weight, self.weight - mint.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
                    )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    ops.assign(
                        self.weight, self.weight + mint.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
                    )
                self.merged = True

    def set_train(self, mode: bool = True) -> None:
        super().set_train(self, mode)
        self._set_train(mode)

    def add_flags(self, **flags):
        self = super().add_flags(**flags)
        training = flags.get("traininig", None)
        if training is None:
            return self
        self._set_train(training)
        return self

    def construct(self, x: Tensor) -> Tensor:
        if self.r > 0 and not self.merged:
            result = self.dense(x, self.weight, self.bias)
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        else:
            result = self.dense(x, self.weight, self.bias)

        return result

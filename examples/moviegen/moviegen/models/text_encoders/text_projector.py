from typing import Type

import mindspore as ms
from mindspore import Tensor, mint, nn

from mindone.models.utils import normal_, zeros_


class TextProjector(nn.Cell):
    def __init__(
        self,
        ul2_in_features: int = 4096,
        metaclip_in_features: int = 1280,
        byt5_in_features: int = 1472,
        out_features: int = 6144,
        layer_norm: Type[nn.Cell] = mint.nn.LayerNorm,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        post_init_weight: bool = True,
        dtype: ms.Type = ms.float32,
    ):
        super().__init__()
        self.ul2_projector = nn.SequentialCell(
            [
                mint.nn.Linear(ul2_in_features, out_features, bias=False, dtype=dtype),
                layer_norm((out_features,), eps=norm_eps, dtype=dtype),
            ]
        )
        self.metaclip_projector = nn.SequentialCell(
            [
                mint.nn.Linear(metaclip_in_features, out_features, bias=False, dtype=dtype),
                layer_norm((out_features,), eps=norm_eps, dtype=dtype),
            ]
        )
        self.byt5_projector = nn.SequentialCell(
            [
                mint.nn.Linear(byt5_in_features, out_features, bias=False, dtype=dtype),
                layer_norm((out_features,), eps=norm_eps, dtype=dtype),
            ]
        )
        self.initializer_range = initializer_range

        # post-init
        if post_init_weight:
            self.initializer_range = initializer_range
            self.init_weights()

    def init_weights(self):
        std = self.initializer_range

        def _init_weights(module):
            if isinstance(module, mint.nn.Linear):
                normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    zeros_(module.weight)

        self.apply(_init_weights)

    def construct(self, ul2_emb: Tensor, metaclip_emb: Tensor, byt5_emb: Tensor) -> Tensor:
        ul2_hidden_states = self.ul2_projector(ul2_emb)
        metaclip_hidden_states = self.metaclip_projector(metaclip_emb)
        byt5_hidden_states = self.byt5_projector(byt5_emb)

        return mint.cat((ul2_hidden_states, metaclip_hidden_states, byt5_hidden_states), dim=1)
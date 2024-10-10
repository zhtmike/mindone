from typing import Optional

from packaging.version import parse

import mindspore as ms
from mindspore import ops

__all__ = [
    "linear",
]

MINDSPORE_VERSION = parse(ms.__version__)


# ================================================================================
# linear
# ================================================================================
def _linear(input: ms.Tensor, weight: ms.Tensor, bias: Optional[ms.Tensor] = None) -> ms.Tensor:
    outputs = ops.matmul(input, weight.T)
    if bias is not None:
        outputs += bias
    return outputs


if MINDSPORE_VERSION >= parse("2.3.0"):
    linear = ms.mint.nn.functional.linear
else:
    linear = _linear

import os
import sys

import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append("./")

from opensora.models.cogvideox.cogvideox_transformer_3d import Rope2D

import mindspore as ms
from mindspore import Tensor


def test_rope():
    ms.set_context(mode=ms.GRAPH_MODE)
    rope = Rope2D(use_internal=False)
    rope_internal = Rope2D(use_internal=True)
    x = Tensor(np.random.uniform(-2, 2, (4, 4, 8192, 128)), dtype=ms.float32)
    freqs_cis = Tensor(np.tile(np.random.uniform(-1, 1, (1, 2, 8192, 128)), (4, 1, 1, 1)), dtype=ms.float32)
    result_0 = rope(x, freqs_cis, text_seq_length=0).asnumpy()
    result_1 = rope_internal(x, freqs_cis, text_seq_length=0).asnumpy()
    np.testing.assert_allclose(result_0, result_1, rtol=1.3e-6, atol=1e-5)


if __name__ == "__main__":
    test_rope()

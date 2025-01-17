import numpy as np

import mindspore as ms
import mindspore.mint as mint
from mindspore import Tensor


def blend_v(a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    blended_b = b.copy()
    for y in range(blend_extent):
        blended_b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
            y / blend_extent
        )
    return blended_b


def blend_v_vec(a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    ratio = mint.arange(blend_extent, dtype=ms.float32) / blend_extent
    ratio = ratio[None, None, None, :, None]
    part, b = mint.split(b, (blend_extent, b.shape[3] - blend_extent), dim=3)
    part = (1 - ratio) * a[:, :, :, -blend_extent:, :] + ratio * part
    b = mint.cat([part, b], dim=3)
    return b


def blend_h(a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
    blend_extent = min(a.shape[4], b.shape[4], blend_extent)
    blended_b = b.copy()
    for x in range(blend_extent):
        blended_b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
            x / blend_extent
        )
    return blended_b


def blend_h_vec(a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
    blend_extent = min(a.shape[4], b.shape[4], blend_extent)
    ratio = mint.arange(blend_extent, dtype=ms.float32) / blend_extent
    ratio = ratio[None, None, None, None, :]
    part, b = mint.split(b, (blend_extent, b.shape[4] - blend_extent), dim=4)
    part = (1 - ratio) * a[:, :, :, :, -blend_extent:] + ratio * part
    b = mint.cat([part, b], dim=4)
    return b


if __name__ == "__main__":
    ms.set_seed(0)
    a = mint.rand(1, 32, 5, 30, 42, dtype=ms.float32)
    b = mint.rand(1, 32, 5, 17, 42, dtype=ms.float32)

    result_0 = blend_v(a, b, 5).asnumpy()
    result_1 = blend_v_vec(a, b, 5).asnumpy()
    np.testing.assert_allclose(result_0, result_1, atol=1e-5)

    a = mint.rand(1, 32, 5, 30, 42, dtype=ms.float32)
    b = mint.rand(1, 32, 5, 30, 29, dtype=ms.float32)

    result_0 = blend_h(a, b, 5).asnumpy()
    result_1 = blend_h_vec(a, b, 5).asnumpy()
    np.testing.assert_allclose(result_0, result_1, atol=1e-5)
    print("Test: Passed.")

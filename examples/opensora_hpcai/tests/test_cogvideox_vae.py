import argparse
import time

import numpy as np
from opensora.models.vae.cogvideox import CogVideoX_VAE, CogVideoX_VAE_Encoder

import mindspore as ms
from mindspore import Tensor

from mindone.data.video_reader import VideoReader
from mindone.utils.seed import set_random_seed


def get_sample_data(path) -> Tensor:
    with VideoReader(path) as reader:
        input_ = reader.fetch_frames(num=17)
    input_ = input_[:, :336, :336]
    input_ = np.transpose(input_, (3, 0, 1, 2))
    input_ = input_.astype(np.float32) / 127.5 - 1
    input_ = Tensor(input_[None], dtype=ms.float32)
    return input_


def test_vae(mode: int, test_full: bool = False):
    ms.set_context(mode=mode)
    # prepare data
    set_random_seed(1024)
    data_0 = get_sample_data("../videocomposer/datasets/webvid5/3.mp4")
    data_1 = get_sample_data("../videocomposer/datasets/webvid5/4.mp4")

    if test_full:
        model = CogVideoX_VAE("models/CogVideoX1.5-5B/vae/diffusion_pytorch_model.safetensors")
    else:
        model = CogVideoX_VAE_Encoder("models/CogVideoX1.5-5B/vae/diffusion_pytorch_model.safetensors")
    output = model(data_0)
    now = time.time()
    output = model(data_1)
    print(f"mode: {mode}, time for forward: {time.time() - now:.3f}s")

    np.save(f"tests/tmp/mode_{mode}.npy", output.asnumpy())


def compare():
    output_graph = np.load("tests/tmp/mode_0.npy")
    output_pynative = np.load("tests/tmp/mode_1.npy")
    np.testing.assert_allclose(output_graph, output_pynative, atol=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=0, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare()
    else:
        test_vae(mode=args.mode, test_full=args.full)
